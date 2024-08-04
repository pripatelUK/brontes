use std::{path::Path, time::Duration};

use brontes_core::decoding::Parser as DParser;
use brontes_database::clickhouse::cex_config::CexDownloadConfig;
use brontes_inspect::Inspectors;
use brontes_metrics::PoirotMetricsListener;
use brontes_types::{
    constants::USDT_ADDRESS_STRING,
    db::cex::{config::CexDexTradeConfig, CexExchange},
    db_write_trigger::{backup_server_heartbeat, start_hr_monitor, HeartRateMonitor},
    init_threadpools, UnboundedYapperReceiver,
};
use clap::Parser;
use tokio::sync::mpsc::unbounded_channel;

use super::{determine_max_tasks, get_env_vars, load_clickhouse, load_database, static_object};
use crate::{
    //banner::rain,
    cli::{get_tracing_provider, init_inspectors, load_tip_database},
    runner::CliContext,
    BrontesRunConfig,
    MevProcessor,
};
const SECONDS_TO_US: u64 = 1_000_000;

#[derive(Debug, Parser)]
pub struct RunArgs {
    /// Optional Start Block, if omitted it will run at tip until killed
    #[arg(long, short)]
    pub start_block:    Option<u64>,
    /// Optional End Block, if omitted it will run historically & at tip until
    /// killed
    #[arg(long, short)]
    pub end_block:      Option<u64>,
    /// Optional Max Tasks, if omitted it will default to 80% of the number of
    /// physical cores on your machine
    #[arg(long, short)]
    pub max_tasks:      Option<u64>,
    /// Optional minimum batch size
    #[arg(long, default_value = "500")]
    pub min_batch_size: u64,
    /// Optional quote asset, if omitted it will default to USDT
    #[arg(long, short, default_value = USDT_ADDRESS_STRING)]
    pub quote_asset:    String,
    /// Inspectors to run. If omitted it defaults to running all inspectors
    #[arg(long, short, value_delimiter = ',')]
    pub inspectors:     Option<Vec<Inspectors>>,

    #[clap(flatten)]
    pub time_window_args:     TimeWindowArgs,
    /// CEX exchanges to consider for cex-dex analysis
    #[arg(
        long,
        short,
        default_value = "Binance,Coinbase,Okex,BybitSpot,Kucoin",
        value_delimiter = ','
    )]
    pub cex_exchanges:        Vec<CexExchange>,
    /// Ensures that dex prices are calculated at every block, even if the
    /// db already contains the price
    #[arg(long, short, default_value = "false")]
    pub force_dex_pricing:    bool,
    /// Turns off dex pricing entirely, inspectors requiring dex pricing won't
    /// calculate USD pnl if we don't have dex pricing in the db & will only
    /// calculate token pnl
    #[arg(long, default_value = "false")]
    pub force_no_dex_pricing: bool,
    /// How many blocks behind chain tip to run.
    #[arg(long, default_value = "10")]
    pub behind_tip:           u64,
    /// Run in CLI only mode (no TUI) - will output progress bars to stdout
    #[arg(long, default_value = "true")]
    pub cli_only:             bool,
    /// Metrics will be exported
    #[arg(long, default_value = "true")]
    pub with_metrics:         bool,
    /// wether or not to use a fallback server.
    #[arg(long, default_value_t = false)]
    pub enable_fallback:      bool,
    /// the address of the fallback server. if the socket breaks,
    /// the fallback server will trigger db writes to ensure we
    /// don't lose data
    #[arg(long)]
    pub fallback_server:      Option<String>,
}

#[derive(Debug, Parser)]
pub struct TimeWindowArgs {
    /// The sliding time window (BEFORE) for cex prices or trades relative to
    /// the block timestamp
    #[arg(long = "tw-before", short = 'b', default_value = "3.0")]
    pub time_window_before:            f64,
    /// The sliding time window (AFTER) for cex prices or trades relative to the
    /// block timestamp
    #[arg(long = "tw-after", short = 'a', default_value = "6.0")]
    pub time_window_after:             f64,
    /// The time window (BEFORE) for cex prices or trades relative to
    /// the block timestamp for fully optimistic calculations
    #[arg(long = "op-tw-before", default_value = "2.0")]
    pub time_window_before_optimistic: f64,
    /// The time window (AFTER) for cex prices or trades relative to
    /// the block timestamp for fully optimistic calculations
    #[arg(long = "op-tw-after", default_value = "5.0")]
    pub time_window_after_optimistic:  f64,
    /// Cex Dex Quotes price time
    #[arg(long = "mk-time", default_value = "0.0")]
    pub quotes_price_time:             f64,
}

impl RunArgs {
    pub async fn execute(
        mut self,
        brontes_db_endpoint: String,
        ctx: CliContext,
    ) -> eyre::Result<()> {
        self.check_proper_range()?;

        let snapshot_mode = !cfg!(feature = "local-clickhouse");
        tracing::info!(%snapshot_mode);

        // Fetch required environment variables.
        let reth_db_path = get_env_vars()?;
        tracing::info!(target: "brontes", "got env vars");
        let quote_asset = self.quote_asset.parse()?;
        tracing::info!(target: "brontes", "parsed quote asset");
        let task_executor = ctx.task_executor;

        let max_tasks = determine_max_tasks(self.max_tasks);
        init_threadpools(max_tasks as usize);

        let (metrics_tx, metrics_rx) = unbounded_channel();
        let metrics_listener = PoirotMetricsListener::new(UnboundedYapperReceiver::new(
            metrics_rx,
            10_000,
            "metrics".to_string(),
        ));

        task_executor.spawn_critical("metrics", metrics_listener);

        let hr = self.try_start_fallback_server().await;

        tracing::info!(target: "brontes", "starting database initialization at: '{}'", brontes_db_endpoint);
        let libmdbx = static_object(load_database(&task_executor, brontes_db_endpoint, hr).await?);

        let tip = static_object(load_tip_database(libmdbx)?);
        tracing::info!(target: "brontes", "initialized libmdbx database");

        let load_window = self.load_time_window();

        let cex_download_config = CexDownloadConfig::new(
            // the run time window. notably we download the max window
            (load_window as u64, load_window as u64),
            self.cex_exchanges.clone(),
        );

        let clickhouse = static_object(load_clickhouse(cex_download_config).await?);
        tracing::info!(target: "brontes", "Databases initialized");

        let only_cex_dex = self
            .inspectors
            .as_ref()
            .map(|f| {
                f.len() == 1 && f.contains(&Inspectors::CexDex)
                    || f.contains(&Inspectors::CexDexMarkout)
            })
            .unwrap_or(false);

        if only_cex_dex {
            self.force_no_dex_pricing = true;
        }

        let trade_config = self.trade_config();

        let inspectors = init_inspectors(
            quote_asset,
            libmdbx,
            self.inspectors,
            self.cex_exchanges,
            trade_config,
            self.with_metrics,
        );

        let tracer =
            get_tracing_provider(Path::new(&reth_db_path), max_tasks, task_executor.clone());
        let parser = static_object(DParser::new(metrics_tx, libmdbx, tracer.clone()).await);

        let executor = task_executor.clone();
        let result = executor
            .clone()
            .spawn_critical_with_graceful_shutdown_signal("run init", |shutdown| async move {
                if let Ok(brontes) = BrontesRunConfig::<_, _, _, MevProcessor>::new(
                    self.start_block,
                    self.end_block,
                    self.behind_tip,
                    max_tasks,
                    self.min_batch_size,
                    quote_asset,
                    self.force_dex_pricing,
                    self.force_no_dex_pricing,
                    inspectors,
                    clickhouse,
                    parser,
                    libmdbx,
                    tip,
                    self.cli_only,
                    self.with_metrics,
                    snapshot_mode,
                    load_window,
                )
                .build(task_executor, shutdown)
                .await
                .map_err(|e| {
                    tracing::error!(%e);
                    e
                }) {
                    brontes.await;
                }
            });

        result.await?;

        Ok(())
    }

    async fn try_start_fallback_server(&self) -> Option<HeartRateMonitor> {
        if self.enable_fallback {
            if let Some(fallback_server) = self.fallback_server.clone() {
                tracing::info!("starting heartbeat");
                backup_server_heartbeat(fallback_server, Duration::from_secs(4)).await;
                None
            } else {
                tracing::info!("starting monitor");
                let (tx, rx) = tokio::sync::mpsc::channel(10);
                if let Err(e) = start_hr_monitor(tx).await {
                    tracing::error!(err=%e);
                }
                tracing::info!("monitor server started");
                Some(HeartRateMonitor::new(Duration::from_secs(7), rx))
            }
        } else {
            None
        }
    }

    /// the time window in seconds for downloading
    fn load_time_window(&self) -> usize {
        self.time_window_args
            .time_window_before
            .max(self.time_window_args.time_window_after)
            .max(self.time_window_args.time_window_before_optimistic)
            .max(self.time_window_args.time_window_after_optimistic) as usize
    }

    fn check_proper_range(&self) -> eyre::Result<()> {
        if let (Some(start), Some(end)) = (&self.start_block, &self.end_block) {
            if start > end {
                return Err(eyre::eyre!("start block must be less than end block"))
            }
        }
        Ok(())
    }

    fn trade_config(&self) -> CexDexTradeConfig {
        CexDexTradeConfig {
            time_window_after_us:  self.time_window_args.time_window_after as u64 * SECONDS_TO_US,
            time_window_before_us: self.time_window_args.time_window_before as u64 * SECONDS_TO_US,
            optimistic_before_us:  self.time_window_args.time_window_before_optimistic as u64
                * SECONDS_TO_US,
            optimistic_after_us:   self.time_window_args.time_window_after_optimistic as u64
                * SECONDS_TO_US,
            quotes_fetch_time:     (self.time_window_args.quotes_price_time * 1000000.0) as u64,
        }
    }
}
