use std::collections::HashMap;

use brontes_types::{
    db::{
        cex::{CexExchange, CexPriceMap, CexQuote},
        redefined_types::{
            malachite::Redefined_Rational,
            primitives::{Redefined_Address, Redefined_Pair},
        },
    },
    pair::Pair,
};
use redefined::{Redefined, RedefinedConvert};
use sorella_db_databases::clickhouse::{self, Row};

use super::{LibmdbxData, ReturnKV};
use crate::libmdbx::CexPrice;

#[derive(Debug, Clone, Row, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CexPriceData {
    pub block_number: u64,
    pub data:         CexPriceMap,
}

impl LibmdbxData<CexPrice> for CexPriceData {
    fn into_key_val(&self) -> ReturnKV<CexPrice> {
        (self.block_number, self.data.clone()).into()
    }
}

#[derive(
    Debug, Clone, serde::Serialize, rkyv::Serialize, rkyv::Deserialize, rkyv::Archive, Redefined,
)]
#[redefined(CexPriceMap)]
#[redefined_attr(
    to_source = "CexPriceMap(self.map.to_source())",
    from_source = "LibmdbxCexPriceMap::new(src.0)"
)]
#[archive(check_bytes)]
pub struct LibmdbxCexPriceMap {
    pub map: HashMap<CexExchange, HashMap<Redefined_Pair, LibmdbxCexQuote>>,
}

impl LibmdbxCexPriceMap {
    fn new(map: HashMap<CexExchange, HashMap<Pair, CexQuote>>) -> Self {
        Self { map: HashMap::from_source(map) }
    }
}

#[derive(
    Debug,
    Clone,
    Hash,
    Eq,
    serde::Serialize,
    rkyv::Serialize,
    rkyv::Deserialize,
    rkyv::Archive,
    Redefined,
)]
#[archive(check_bytes)]
#[redefined(CexQuote)]
pub struct LibmdbxCexQuote {
    pub exchange:  CexExchange,
    pub timestamp: u64,
    pub price:     (Redefined_Rational, Redefined_Rational),
    pub token0:    Redefined_Address,
}

impl PartialEq for LibmdbxCexQuote {
    fn eq(&self, other: &Self) -> bool {
        self.clone().to_source().eq(&other.clone().to_source())
    }
}
