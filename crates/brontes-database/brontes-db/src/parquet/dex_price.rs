use std::sync::Arc;

use arrow::{
    array::{
        Array, BooleanBuilder, Float64Array, Float64Builder, StringArray, StringBuilder,
        UInt16Builder, UInt64Builder,
    },
    datatypes::{DataType, Field, Schema},
    error::ArrowError,
    record_batch::RecordBatch,
};
use brontes_types::{db::dex::DexQuoteWithIndex, pair::Pair};
use malachite::num::conversion::traits::RoundingFrom;
use malachite_base::{num::conversion::traits::RoundingFrom, rounding_modes::RoundingMode};
use tracing::warn;

use super::utils::build_record_batch;

/// Converts a vector of DexQuoteWithIndex (representing quotes for different tx
/// indices within potentially multiple blocks) into a flattened Arrow
/// RecordBatch.
///
/// The input format assumes a structure like `Vec<(block_number,
/// DexQuoteWithIndex)>`.
pub fn dex_quotes_to_record_batch(
    block_quotes: Vec<(u64, DexQuoteWithIndex)>,
) -> Result<RecordBatch, ArrowError> {
    // Estimate initial capacity (can be refined)
    let initial_capacity = block_quotes.iter().map(|(_, dq)| dq.quote.len()).sum();

    let mut block_number_builder = UInt64Builder::with_capacity(initial_capacity);
    let mut tx_idx_builder = UInt16Builder::with_capacity(initial_capacity);
    let mut pair_token0_builder = StringBuilder::new(); // Capacity estimated later
    let mut pair_token1_builder = StringBuilder::new();
    let mut pre_state_price_builder = Float64Builder::with_capacity(initial_capacity);
    let mut post_state_price_builder = Float64Builder::with_capacity(initial_capacity);
    let mut pool_liquidity_builder = Float64Builder::with_capacity(initial_capacity);
    let mut goes_through_token0_builder = StringBuilder::new();
    let mut goes_through_token1_builder = StringBuilder::new();
    let mut is_transfer_builder = BooleanBuilder::with_capacity(initial_capacity);
    let mut first_hop_connections_builder = UInt64Builder::with_capacity(initial_capacity);

    for (block_number, dex_quote_with_index) in block_quotes {
        let tx_idx = dex_quote_with_index.tx_idx;
        for (pair, dex_prices) in dex_quote_with_index.quote {
            block_number_builder.append_value(block_number);
            tx_idx_builder.append_value(tx_idx as u16);
            pair_token0_builder.append_value(pair.0.to_string());
            pair_token1_builder.append_value(pair.1.to_string());

            let (pre_state_f64, _) =
                f64::rounding_from(&dex_prices.pre_state, RoundingMode::Nearest);
            pre_state_price_builder.append_value(pre_state_f64);

            let (post_state_f64, _) =
                f64::rounding_from(&dex_prices.post_state, RoundingMode::Nearest);
            post_state_price_builder.append_value(post_state_f64);

            let (pool_liquidity_f64, _) =
                f64::rounding_from(&dex_prices.pool_liquidity, RoundingMode::Nearest);
            pool_liquidity_builder.append_value(pool_liquidity_f64);

            goes_through_token0_builder.append_value(dex_prices.goes_through.0.to_string());
            goes_through_token1_builder.append_value(dex_prices.goes_through.1.to_string());
            is_transfer_builder.append_value(dex_prices.is_transfer);
            first_hop_connections_builder.append_value(dex_prices.first_hop_connections as u64);
        }
    }

    let schema = Schema::new(vec![
        Field::new("block_number", DataType::UInt64, false),
        Field::new("tx_idx", DataType::UInt16, false),
        Field::new("pair_token0_address", DataType::Utf8, false),
        Field::new("pair_token1_address", DataType::Utf8, false),
        // Prices are still nullable because NaN is a valid Float64 value, not a true null absence
        // of value. Parquet/Arrow handles NaN correctly within Float64 columns.
        Field::new("pre_state_price", DataType::Float64, true),
        Field::new("post_state_price", DataType::Float64, true),
        Field::new("pool_liquidity", DataType::Float64, true),
        Field::new("goes_through_token0_address", DataType::Utf8, false),
        Field::new("goes_through_token1_address", DataType::Utf8, false),
        Field::new("is_transfer", DataType::Boolean, false),
        Field::new("first_hop_connections", DataType::UInt64, false),
    ]);

    build_record_batch(
        schema,
        vec![
            Arc::new(block_number_builder.finish()),
            Arc::new(tx_idx_builder.finish()),
            Arc::new(pair_token0_builder.finish()),
            Arc::new(pair_token1_builder.finish()),
            Arc::new(pre_state_price_builder.finish()),
            Arc::new(post_state_price_builder.finish()),
            Arc::new(pool_liquidity_builder.finish()),
            Arc::new(goes_through_token0_builder.finish()),
            Arc::new(goes_through_token1_builder.finish()),
            Arc::new(is_transfer_builder.finish()),
            Arc::new(first_hop_connections_builder.finish()),
        ],
    )
}
