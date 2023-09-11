SELECT tx_hash
FROM
(
    SELECT arrayJoin(transaction_hashes) AS tx_hash
    FROM blocks
    WHERE (block_number = ?) AND (block_hash = ?) AND (valid = 1)
) AS subquery
WHERE tx_hash NOT IN (
    SELECT tx_hash
    FROM unique_mempool
)