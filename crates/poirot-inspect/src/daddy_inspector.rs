use std::{sync::Arc, task::Poll};

use poirot_labeller::Metadata;
use poirot_types::{
    classified_mev::{ClassifiedMev, MevBlock, SpecificMev},
    normalized_actions::Actions,
    tree::TimeTree
};
use reth_primitives::Address;
use reth_rpc_types::trace::parity::Action;

type InspectorFut<'a> =
    JoinAll<Pin<Box<dyn Future<Output = Vec<(ClassifiedMev, Box<dyn SpecificMev>)>> + Send + 'a>>>;

pub type DaddyInspectorResults = (MevBlock, Vec<(ClassifiedMev, Box<dyn SpecificMev>)>);

pub struct BlockPreprocessing {
    meta_data:           Arc<Metadata>,
    cumulative_gas_used: u64,
    cumulative_gas_paid: u64,
    builder_address:     Address
}

pub struct DaddyInspector<'a, const N: usize> {
    baby_inspectors:      &'a [&'a Box<dyn Inspector<Mev = Box<dyn SpecificMev>>>; N],
    inspectors_execution: Option<InspectorFut<'a>>,
    pre_processing:       Option<BlockPreprocessing>
}

impl<'a, const N: usize> DaddyInspector<'a, N> {
    pub fn new(baby_inspectors: &'a [&'a Box<dyn Inspector<Mev = dyn SpecificMev>>; N]) -> Self {
        Self { baby_inspectors, inspectors_execution: None, pre_processing: None }
    }

    pub fn is_processing(&self) -> bool {
        self.inspectors_execution.is_some()
    }

    pub fn on_new_tree(&mut self, tree: Arc<TimeTree<Actions>>, meta_data: Arc<Metadata>) {
        self.inspectors_execution = Some(join_all(
            self.baby_inspectors
                .iter()
                .map(|inspector| inspector.process_tree(tree.clone(), metadata.clone()))
        ) as InspectorFut<'a>);

        self.pre_process(tree, meta_data);
    }

    fn pre_process(&mut self, tree: Arc<TimeTree<Actions>>, meta_data: Arc<Metadata>) {
        let builder_address = tree.header.beneficiary;
        let cumulative_gas_used = tree
            .roots
            .iter()
            .map(|root| root.gas_details.gas_used)
            .sum::<u64>();

        let cumulative_gas_paid = tree
            .roots
            .iter()
            .map(|root| root.gas_details.effective_gas_price * root.gas_details.gas_used)
            .sum::<u64>();

        self.pre_processing = Some(BlockPreprocessing {
            meta_data,
            cumulative_gas_used,
            cumulative_gas_paid,
            builder_address
        });
    }

    fn on_baby_resolution(
        &mut self,
        baby_data: Vec<Vec<(ClassifiedMev, Box<dyn SpecificMev>)>>
    ) -> Poll<Option<DaddyInspectorResults>> {
        let pre_processing = self.pre_processing.take().unwrap();

        let cum_mev_priority_fee_paid = baby_data
            .iter()
            .flatten()
            .map(|(_, mev)| mev.priority_fee_paid())
            .sum::<u64>();

        let builder_eth_profit = (total_bribe + pre_processing.cumulative_gas_paid);

        let mut mev_block = MevBlock {
            block_hash: pre_processing.meta_data.block_hash,
            block_number: pre_processing.meta_data.block_num,
            mev_count: baby_data.iter().flatten().count() as u64,
            submission_eth_price: pre_processing.meta_data.eth_prices.0,
            finalized_eth_price: pre_processing.meta_data.eth_prices.1,
            cumulative_gas_used: pre_processing.cumulative_gas_used,
            cumulative_gas_paid: pre_processing.cumulative_gas_paid,
            total_bribe: baby_data
                .iter()
                .flatten()
                .map(|(_, mev)| mev.bribe())
                .sum::<u64>(),
            cumulative_mev_priority_fee_paid: cum_mev_priority_fee_paid,
            builder_address: pre_processing.builder_address,
            builder_eth_profit,
            builder_submission_profit_usd: builder_eth_profit
                * pre_processing.meta_data.eth_prices.0,
            builder_finalized_profit_usd: builder_eth_profit
                * pre_processing.meta_data.eth_prices.1,
            proposer_fee_recipient: pre_processing.meta_data.proposer_fee_recipient,
            proposer_mev_reward: pre_processing.meta_data.proposer_mev_reward,
            proposer_submission_mev_reward_usd: pre_processing.meta_data.proposer_mev_reward
                * pre_processing.meta_data.eth_prices.0,
            proposer_finalized_mev_reward_usd: pre_processing.meta_data.proposer_mev_reward
                * pre_processing.meta_data.eth_prices.1,
            cumulative_mev_submission_profit_usd: cum_mev_priority_fee_paid
                * pre_processing.meta_data.eth_prices.0,
            cumulative_mev_finalized_profit_usd: cum_mev_priority_fee_paid
                * pre_processing.meta_data.eth_prices.1
        };
    }
}

impl<const N: usize> Stream for DaddyInspector<'_, N> {
    type Item = DaddyInspectorResults;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(mut calculations) = self.inspectors_execution.take() {
            match calculations.poll_next_unpin(cx) {
                Poll::Ready(data) => self.on_baby_resolution(data),
                Poll::Pending => {
                    self.inspectors_execution = Some(calculations);
                    Poll::Pending
                }
            }
        }
    }
}
