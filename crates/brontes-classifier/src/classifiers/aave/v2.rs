use alloy_primitives::{Address, U256};
use brontes_database_libmdbx::{implementation::tx::LibmdbxTx, tables::AddressToTokens};
use brontes_macros::{action_dispatch, action_impl};
use brontes_types::normalized_actions::{NormalizedFlashLoan, NormalizedLiquidation};
use reth_db::{mdbx::RO, transaction::DbTx};

use crate::AaveV2::{flashLoanCall, liquidationCallCall};

action_impl!(
    LiquidationCallImplV2,
    Liquidation,
    liquidationCallCall,
    [],
    AaveV2,
    call_data: true,
    |trace_index,
    from_address: Address,
    target_address: Address,
    call_data: liquidationCallCall,
    db_tx: &LibmdbxTx<RO>| {
        return Some(NormalizedLiquidation {
            trace_index,
            pool: target_address,
            liquidator: from_address,
            debtor: call_data.user,
            collateral_asset: call_data.collateralAsset,
            debt_asset: call_data.debtAsset,
            covered_debt: call_data.debtToCover,
            // filled in later
            liquidated_collateral: U256::ZERO,
        })
    }
);

action_impl!(
    FlashloanImplV2,
    FlashLoan,
    flashLoanCall,
    [],
    AaveV2,
    call_data: true,
    |trace_index,
    from_address: Address,
    target_address: Address,
    call_data: flashLoanCall,
    db_tx: &LibmdbxTx<RO> | {

        let tokens = db_tx.get::<AddressToTokens>(target_address).ok()??;
        let [mut token_0, mut token_1] = [tokens.token0, tokens.token1];

        return Some(NormalizedFlashLoan {
            trace_index,
            from: from_address,
            pool: target_address,
            receiver_contract: call_data.receiverAddress,
            assets: call_data.assets,
            amounts: call_data.amounts,
            aave_mode: Some((call_data.modes, call_data.onBehalfOf)),
            // Set to zero at this stage, will be calculated upon finalized classification
            child_actions: vec![],
            repayments: vec![],
            fees_paid: vec![],
        })

    }


);

action_dispatch!(AaveV2Classifier, LiquidationCallImplV2, FlashloanImplV2);
