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
use malachite::{
    num::conversion::{string::options::ToSciOptions, traits::ToSci},
    Rational,
};
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

    // Configure options for scientific notation string conversion
    let mut sci_options = ToSciOptions::default();
    // Using max_significant_digits is generally better for preserving overall
    // magnitude
    sci_options.set_max_significant_digits(Some(36));

    for (block_number, dex_quote_with_index) in block_quotes {
        let tx_idx = dex_quote_with_index.tx_idx;
        for (pair, dex_prices) in dex_quote_with_index.quote {
            block_number_builder.append_value(block_number);
            tx_idx_builder.append_value(tx_idx as u16);
            pair_token0_builder.append_value(pair.0.to_string());
            pair_token1_builder.append_value(pair.1.to_string());

            // Convert Rational to f64 using to_sci_with_options and parse, fallback to NaN
            match dex_prices
                .pre_state
                .to_sci_with_options(sci_options)
                .parse::<f64>()
            {
                Ok(val) => pre_state_price_builder.append_value(val),
                Err(e) => {
                    warn!(target: "brontes::db::export::dex_price", block=block_number, tx_idx=tx_idx, pair=?pair, field="pre_state", value=%dex_prices.pre_state, error=?e, "Failed to parse Rational as f64, using NaN.");
                    pre_state_price_builder.append_value(f64::NAN);
                }
            }
            match dex_prices
                .post_state
                .to_sci_with_options(sci_options)
                .parse::<f64>()
            {
                Ok(val) => post_state_price_builder.append_value(val),
                Err(e) => {
                    warn!(target: "brontes::db::export::dex_price", block=block_number, tx_idx=tx_idx, pair=?pair, field="post_state", value=%dex_prices.post_state, error=?e, "Failed to parse Rational as f64, using NaN.");
                    post_state_price_builder.append_value(f64::NAN);
                }
            }
            match dex_prices
                .pool_liquidity
                .to_sci_with_options(sci_options)
                .parse::<f64>()
            {
                Ok(val) => pool_liquidity_builder.append_value(val),
                Err(e) => {
                    warn!(target: "brontes::db::export::dex_price", block=block_number, tx_idx=tx_idx, pair=?pair, field="pool_liquidity", value=%dex_prices.pool_liquidity, error=?e, "Failed to parse Rational as f64, using NaN.");
                    pool_liquidity_builder.append_value(f64::NAN);
                }
            }

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
