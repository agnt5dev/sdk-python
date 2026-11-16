//! A delivery boundary for a run's queued stream events.
//!
//! Hosted agent streams queue their events in sdk-core's journal queue, and a
//! periodic sender delivers them. Before the stream's next durable checkpoint,
//! the run's queued events must be on their way, or the checkpoint could
//! become visible first.
//!
//! The sender takes each run's flush barrier before draining that run's events
//! and holds it until they are acknowledged, and durable checkpoints for the
//! run queue behind the same barrier. So once the run has nothing left in the
//! queue, its next checkpoint cannot overtake those events, and there is
//! nothing more to wait for.

use std::time::Duration;

use agnt5_sdk_core::JournalEventQueue;

/// How long to wait before giving up on a stalled sender. This was the old
/// fixed wait, so the boundary is never slower than it used to be.
pub(crate) const STREAM_DELIVERY_CEILING: Duration = Duration::from_millis(200);

/// How often to check the queue. The sender drains on its own interval, so
/// this only bounds how late the drain is noticed.
const POLL_INTERVAL: Duration = Duration::from_millis(5);

/// Waits until `run_id` has no events left in `queue`, up to `ceiling`.
/// Returns whether the run's events left the queue in time.
///
/// A held run returns at once: the periodic sender never drains it, and its
/// events ride the run's completion request, which waits for them itself.
pub(crate) async fn wait_for_stream_delivery(
    queue: &JournalEventQueue,
    run_id: &str,
    ceiling: Duration,
) -> bool {
    let deadline = tokio::time::Instant::now() + ceiling;
    loop {
        if queue.is_held(run_id) || !queue.contains_run(run_id) {
            return true;
        }
        if tokio::time::Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agnt5_sdk_core::{JournalEventMessage, JournalQueueConfig};
    use std::time::Instant;

    fn queue_with(run_ids: &[&str]) -> JournalEventQueue {
        let queue = JournalEventQueue::new(JournalQueueConfig::default());
        for run_id in run_ids {
            queue
                .push(JournalEventMessage::new(
                    run_id.to_string(),
                    "lm.message.stop".to_string(),
                    Vec::new(),
                ))
                .unwrap();
        }
        queue
    }

    #[tokio::test]
    async fn returns_at_once_when_the_run_has_nothing_queued() {
        // Another run's backlog must not delay this one.
        let queue = queue_with(&["other-run"]);
        let started = Instant::now();
        assert!(wait_for_stream_delivery(&queue, "run", STREAM_DELIVERY_CEILING).await);
        assert!(started.elapsed() < Duration::from_millis(50));
    }

    #[tokio::test]
    async fn returns_once_the_sender_drains_the_run() {
        let queue = queue_with(&["run"]);
        let sender = queue.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            sender.drain_run_events("run");
        });
        let started = Instant::now();
        assert!(wait_for_stream_delivery(&queue, "run", STREAM_DELIVERY_CEILING).await);
        let waited = started.elapsed();
        assert!(waited >= Duration::from_millis(20));
        assert!(waited < Duration::from_millis(150), "waited {waited:?}");
    }

    #[tokio::test]
    async fn returns_at_once_for_a_held_run() {
        let queue = queue_with(&["run"]);
        queue.hold_run("run");
        let started = Instant::now();
        assert!(wait_for_stream_delivery(&queue, "run", STREAM_DELIVERY_CEILING).await);
        assert!(started.elapsed() < Duration::from_millis(50));
    }

    #[tokio::test]
    async fn gives_up_at_the_ceiling_when_the_sender_stalls() {
        let queue = queue_with(&["run"]);
        let started = Instant::now();
        assert!(!wait_for_stream_delivery(&queue, "run", Duration::from_millis(30)).await);
        assert!(started.elapsed() >= Duration::from_millis(30));
    }
}
