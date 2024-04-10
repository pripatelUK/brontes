use std::path::PathBuf;

use color_eyre::eyre::Result;
use directories::ProjectDirs;
use lazy_static::lazy_static;
use tracing::error;
use tracing_error::ErrorLayer;
use tracing_subscriber::{
    prelude::__tracing_subscriber_SubscriberExt, util::SubscriberInitExt, Layer,
};

pub fn get_config_dir() -> PathBuf {
    let directory = PathBuf::from(".").join("config");
    tracing::info!("Config directory: {:?}", directory);
    directory
}

#[macro_export]
macro_rules! get_symbols_from_transaction_accounting {
    ($data:expr) => {{
        use brontes_types::{db::token_info::TokenInfoWithAddress, hasher::FastHashSet};

        let mut token_info_with_addresses: Vec<TokenInfoWithAddress> = Vec::new();
        for transaction in $data {
            for address_delta in &transaction.address_deltas {
                for token_delta in &address_delta.token_deltas {
                    token_info_with_addresses.push(token_delta.token.clone());
                }
            }
        }
        let mut symbols = FastHashSet::default();
        let unique_symbols: Vec<String> = token_info_with_addresses
            .iter()
            .filter_map(|x| {
                let symbol = x.inner.symbol.to_string();
                if symbols.insert(symbol.clone()) {
                    Some(symbol)
                } else {
                    None
                }
            })
            .collect();

        unique_symbols.join(", ")
    }};
}
