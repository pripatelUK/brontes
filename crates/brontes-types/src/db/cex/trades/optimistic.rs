use std::{f64::consts::E, fmt::Display, ops::Mul};

use alloy_primitives::FixedBytes;
use itertools::Itertools;
use malachite::{
    num::basic::traits::{One, Zero},
    Rational,
};

use super::config::CexDexTradeConfig;
use crate::{
    constants::{USDC_ADDRESS, USDT_ADDRESS},
    db::cex::{
        trades::{
            utils::{log_insufficient_trade_volume, log_missing_trade_data, TimeBasketQueue},
            CexTrades, Direction, SortedTrades,
        },
        CexExchange,
    },
    display::utils::format_etherscan_url,
    mev::OptimisticTrade,
    normalized_actions::NormalizedSwap,
    pair::Pair,
    utils::ToFloatNearest,
    FastHashMap,
};

pub const BASE_EXECUTION_QUALITY: usize = 70;

const PRE_SCALING_DIFF: u64 = 200_000;
const TIME_STEP: u64 = 100_000;

/// the calculated price based off of trades with the estimated exchanges with
/// volume amount that where used to hedge
#[derive(Debug, Clone)]
pub struct ExchangePrice {
    // cex exchange with amount of volume executed on it
    pub trades_used: Vec<OptimisticTrade>,
    /// the pairs that were traded through in order to get this price.
    /// in the case of a intermediary, this will be 2, otherwise, 1
    pub pairs:       Vec<Pair>,
    pub final_price: Rational,
}

impl Mul for ExchangePrice {
    type Output = ExchangePrice;

    fn mul(mut self, rhs: Self) -> Self::Output {
        self.pairs.extend(rhs.pairs);
        self.final_price *= rhs.final_price;
        self.trades_used.extend(rhs.trades_used);

        self
    }
}

impl Display for ExchangePrice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{:#?}", self.trades_used)?;
        writeln!(f, "{}", self.final_price.clone().to_float())
    }
}

pub type MakerTaker = (ExchangePrice, ExchangePrice);

impl<'a> SortedTrades<'a> {
    // Calculates VWAPs for the given pair across all provided exchanges - this
    // will assess trades across each exchange
    //
    // For non-intermediary dependent pairs, we do the following:
    // - 1. Adjust each exchange's trade set by the assumed execution quality for
    //   the given pair on the exchange. We assess a larger percentage of trades if
    //   execution quality is assumed to be lower.
    // - 2. Exclude trades with a volume that is too large to be considered
    //   potential hedging trades.
    // - 3. Order all trades for each exchange by price.
    // - 4. Finally, we pick a vector of trades whose total volume is closest to the
    //   swap volume.
    // - 5. Calculate the VWAP for the chosen set of trades.

    // For non-intermediary dependant pairs
    // - 1. Calculate VWAPs for all potential intermediary pairs (using above
    //   process)
    // -- Pair's with insufficient volume will be filtered out here which will
    // filter the route in the next step
    // - 2. Combines VWAP's to assess potential routes
    // - 3. Selects most profitable route and returns it as the Price
    // -- It should be noted here that this will not aggregate multiple possible
    // routes
    pub(crate) fn get_optimistic_price(
        &mut self,
        config: CexDexTradeConfig,
        _exchanges: &[CexExchange],
        block_timestamp: u64,
        pair: Pair,
        volume: &Rational,
        quality: Option<FastHashMap<CexExchange, FastHashMap<Pair, usize>>>,
        bypass_vol: bool,
        dex_swap: &NormalizedSwap,
        tx_hash: FixedBytes<32>,
    ) -> Option<MakerTaker> {
        if pair.0 == pair.1 {
            return Some((
                ExchangePrice {
                    trades_used: vec![],
                    pairs:       vec![pair],
                    final_price: Rational::ONE,
                },
                ExchangePrice {
                    trades_used: vec![],
                    pairs:       vec![pair],
                    final_price: Rational::ONE,
                },
            ))
        }

        let res = self
            .get_optimistic_direct(
                config,
                block_timestamp,
                pair,
                volume,
                bypass_vol,
                quality.as_ref(),
                dex_swap,
                tx_hash,
            )
            .or_else(|| {
                self.get_optimistic_via_intermediary(
                    config,
                    block_timestamp,
                    pair,
                    volume,
                    bypass_vol,
                    quality.as_ref(),
                    dex_swap,
                    tx_hash,
                )
            });

        if res.is_none() {
            tracing::debug!(target: "brontes_types::db::cex::optimistic", ?pair, "No price VMAP found for {}-{} in optimistic time window. \n Tx: {}", dex_swap.token_in.symbol, dex_swap.token_out.symbol, format_etherscan_url(&tx_hash));
        }

        res
    }

    fn get_optimistic_via_intermediary(
        &self,
        config: CexDexTradeConfig,
        block_timestamp: u64,
        pair: Pair,
        volume: &Rational,
        bypass_vol: bool,
        quality: Option<&FastHashMap<CexExchange, FastHashMap<Pair, usize>>>,
        dex_swap: &NormalizedSwap,
        tx_hash: FixedBytes<32>,
    ) -> Option<MakerTaker> {
        self.calculate_intermediary_addresses(&pair)
            .into_iter()
            .filter_map(|intermediary| {
                let pair0 = Pair(pair.0, intermediary);
                let pair1 = Pair(intermediary, pair.1);

                tracing::debug!(target: "brontes_types::db::cex::trades::optimistic", ?pair, ?intermediary, "trying via intermediary");

                let mut bypass_intermediary_vol = false;

                // bypass volume requirements for stable pairs
                if pair0.0 == USDC_ADDRESS && pair0.1 == USDT_ADDRESS
                || pair0.0 == USDT_ADDRESS && pair0.1 == USDC_ADDRESS {
                    bypass_intermediary_vol = true;
                }


                let first_leg = self.get_optimistic_direct(
                    config,
                    block_timestamp,
                    pair0,
                    volume,
                    bypass_vol || bypass_intermediary_vol,
                    quality,
                    dex_swap,
                    tx_hash,
                )?;
                let new_vol = volume * &first_leg.0.final_price;

                bypass_intermediary_vol = false;
                if pair1.0 == USDT_ADDRESS && pair1.1 == USDC_ADDRESS
                || pair1.0 == USDC_ADDRESS && pair1.1 == USDT_ADDRESS{
                    bypass_intermediary_vol = true;
                }

                let second_leg = self.get_optimistic_direct(
                    config,
                    block_timestamp,
                    pair1,
                    &new_vol,
                    bypass_vol || bypass_intermediary_vol,
                    quality,
                    dex_swap,
                    tx_hash,
                )?;


                let maker = first_leg.0  * second_leg.0;
                let taker = first_leg.1 * second_leg.1;


                Some((maker, taker))
            })
            .max_by_key(|a| a.0.final_price.clone())
    }

    fn get_optimistic_direct(
        &self,
        config: CexDexTradeConfig,
        block_timestamp: u64,
        pair: Pair,
        volume: &Rational,
        bypass_vol: bool,
        quality: Option<&FastHashMap<CexExchange, FastHashMap<Pair, usize>>>,
        dex_swap: &NormalizedSwap,
        tx_hash: FixedBytes<32>,
    ) -> Option<MakerTaker> {
        // Populate Map of Assumed Execution Quality by Exchange
        // - We're making the assumption that the stat arber isn't hitting *every* good
        //   markout for each pair on each exchange.
        // - Quality percent adjusts the total percent of "good" trades the arber is
        //   capturing for the relevant pair on a given exchange.

        let quality_pct = quality.map(|map| {
            map.iter()
                .map(|(k, v)| (*k, v.get(&pair).copied().unwrap_or(BASE_EXECUTION_QUALITY)))
                .collect::<FastHashMap<_, _>>()
        });

        let trade_data = self.get_trades(pair, dex_swap, tx_hash)?;

        let mut baskets_queue = TimeBasketQueue::new(trade_data, block_timestamp, quality_pct);

        baskets_queue.construct_time_baskets();

        while baskets_queue.volume.lt(volume) {
            if baskets_queue.get_min_time_delta(block_timestamp) >= config.optimistic_before_us
                || baskets_queue.get_max_time_delta(block_timestamp) >= config.optimistic_after_us
            {
                break
            }

            let min_expand = (baskets_queue.get_max_time_delta(block_timestamp)
                >= PRE_SCALING_DIFF)
                .then_some(TIME_STEP)
                .unwrap_or_default();

            baskets_queue.expand_time_bounds(min_expand, TIME_STEP);
        }

        let mut trades_used: Vec<CexTrades> = Vec::new();
        let mut unfilled = Rational::ZERO;

        // This pushed the unfilled to the next basket, given how we create the baskets
        // this means we will start from the baskets closest to the block time
        for basket in baskets_queue.baskets {
            let to_fill: Rational = ((&basket.volume / &baskets_queue.volume) * volume) + &unfilled;

            let (basket_trades, basket_unfilled) = basket.get_trades_used(&to_fill);

            unfilled = basket_unfilled;
            trades_used.extend(basket_trades);
        }

        let mut vxp_maker = Rational::ZERO;
        let mut vxp_taker = Rational::ZERO;
        let mut trade_volume = Rational::ZERO;
        let mut trade_volume_weight = Rational::ZERO;

        let mut optimistic_trades = Vec::with_capacity(trades_used.len());

        for trade in trades_used {
            let weight = calculate_weight(block_timestamp, trade.timestamp);

            let (m_fee, t_fee) = trade.exchange.fees();

            vxp_maker += (&trade.price * (Rational::ONE - m_fee)) * &trade.amount * &weight;
            vxp_taker += (&trade.price * (Rational::ONE - t_fee)) * &trade.amount * &weight;
            trade_volume_weight += &trade.amount * &weight;
            trade_volume += &trade.amount;

            optimistic_trades.push(OptimisticTrade {
                volume: trade.amount.clone(),
                pair,
                price: trade.price.clone(),
                exchange: trade.exchange,
                timestamp: trade.timestamp,
            });
        }

        if &trade_volume < volume && !bypass_vol {
            log_insufficient_trade_volume(pair, dex_swap, &tx_hash, trade_volume, volume.clone());
            return None
        } else if trade_volume == Rational::ZERO {
            return None
        }

        let maker = ExchangePrice {
            trades_used: optimistic_trades.clone(),
            pairs:       vec![pair],
            final_price: vxp_maker / &trade_volume_weight,
        };

        let taker = ExchangePrice {
            trades_used: optimistic_trades,
            pairs:       vec![pair],
            final_price: vxp_taker / &trade_volume_weight,
        };

        Some((maker, taker))
    }

    pub fn get_trades(
        &'a self,
        pair: Pair,
        dex_swap: &NormalizedSwap,
        tx_hash: FixedBytes<32>,
    ) -> Option<OptimisticTradeData> {
        if let Some((indices, trades)) = self.0.get(&pair) {
            let adjusted_trades = trades
                .iter()
                .map(|trade| trade.adjust_for_direction(Direction::Sell))
                .collect_vec();

            Some(OptimisticTradeData {
                indices:   *indices,
                trades:    adjusted_trades,
                direction: Direction::Sell,
            })
        } else {
            let flipped_pair = pair.flip();

            if let Some((indices, trades)) = self.0.get(&flipped_pair) {
                let adjusted_trades = trades
                    .iter()
                    .map(|trade| trade.adjust_for_direction(Direction::Buy))
                    .collect_vec();

                Some(OptimisticTradeData {
                    indices:   *indices,
                    trades:    adjusted_trades,
                    direction: Direction::Buy,
                })
            } else {
                log_missing_trade_data(dex_swap, &tx_hash);
                None
            }
        }
    }
}

pub struct Trades<'a> {
    pub trades:    Vec<(CexExchange, Vec<&'a CexTrades>)>,
    pub direction: Direction,
}

pub struct OptimisticTradeData {
    pub indices:   (usize, usize),
    pub trades:    Vec<CexTrades>,
    pub direction: Direction,
}

const PRE_DECAY: f64 = -0.0000003;
const POST_DECAY: f64 = -0.00000012;

/// Calculates the weight for a trade using a bi-exponential decay function
/// based on its timestamp relative to a block time.
///
/// This function is designed to account for the risk associated with the timing
/// of trades in relation to block times in the context of cex-dex
/// arbitrage. This assumption underpins our pricing model: trades that
/// occur further from the block time are presumed to carry higher uncertainty
/// and an increased risk of adverse market conditions potentially impacting
/// arbitrage outcomes. Accordingly, the decay rates (`PRE_DECAY` for pre-block
/// and `POST_DECAY` for post-block) adjust the weight assigned to each trade
/// based on its temporal proximity to the block time.
///
/// Trades after the block are assumed to be generally preferred by arbitrageurs
/// as they have confirmation that their DEX swap is executed. However, this
/// preference can vary for less competitive pairs where the opportunity and
/// timing of execution might differ.
///
/// # Parameters
/// - `block_time`: The timestamp of the block as seen first on the peer-to-peer
///   network.
/// - `trade_time`: The timestamp of the trade to be weighted.
///
/// # Returns
/// Returns a `Rational` representing the calculated weight for the trade. The
/// weight is determined by:
/// - `exp(-PRE_DECAY * (block_time - trade_time))` for trades before the block
///   time.
/// - `exp(-POST_DECAY * (trade_time - block_time))` for trades after the block
///   time.

fn calculate_weight(block_time: u64, trade_time: u64) -> Rational {
    let pre = trade_time < block_time;

    Rational::try_from_float_simplest(if pre {
        E.powf(PRE_DECAY * (block_time - trade_time) as f64)
    } else {
        E.powf(POST_DECAY * (trade_time - block_time) as f64)
    })
    .unwrap()
}
