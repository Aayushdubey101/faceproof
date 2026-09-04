"""Blockchain anchoring on Base Sepolia.

The anchor is a 0-value self-transaction carrying the 32-byte SHA-256 digest as
calldata. No Solidity, no deploy step, no ABI - and the record is still public,
immutable, timestamped by block and readable from any node or explorer.

ponytail: a calldata anchor is not queryable by subject; a verifier must already
hold the transaction hash. Add a registry contract with indexed records only if
lookup-by-person becomes a requirement.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from faceproof.evidence.hashing import DIGEST_BYTES

NETWORK_NAME = "Base Sepolia"
CHAIN_ID = 84532
DEFAULT_RPC_URL = "https://sepolia.base.org"


def connect(rpc_url: Optional[str] = None) -> Any:
    """Open an RPC connection. `web3` is imported lazily - `verify` is cheap."""
    from web3 import Web3  # deferred: keeps the rest of the pipeline importable without it

    rpc_url = rpc_url or os.environ.get("BLOCKCHAIN_RPC_URL") or DEFAULT_RPC_URL

    web3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 60}))
    if not web3.is_connected():
        raise RuntimeError(f"cannot reach RPC endpoint {rpc_url}")
    return web3


def anchor(
    digest_hex: str,
    rpc_url: Optional[str] = None,
    private_key: Optional[str] = None,
) -> dict:
    """Write the evidence fingerprint to the chain. Returns the receipt details."""
    digest = bytes.fromhex(digest_hex)
    if len(digest) != DIGEST_BYTES:
        raise ValueError(f"expected a {DIGEST_BYTES}-byte sha256 digest, got {len(digest)}")

    private_key = private_key or os.environ.get("BLOCKCHAIN_PRIVATE_KEY")
    if not private_key:
        raise RuntimeError("BLOCKCHAIN_PRIVATE_KEY is not set (copy .env.example to .env)")
    if not private_key.startswith("0x"):
        private_key = "0x" + private_key

    web3 = connect(rpc_url)
    account = web3.eth.account.from_key(private_key)

    transaction = {
        "from": account.address,
        "to": account.address,
        "value": 0,
        "data": digest,
        "nonce": web3.eth.get_transaction_count(account.address),
        "chainId": web3.eth.chain_id,
        "gasPrice": web3.eth.gas_price,
    }
    transaction["gas"] = web3.eth.estimate_gas(transaction)

    signed = account.sign_transaction(transaction)
    tx_hash = web3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = web3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)

    if receipt["status"] != 1:
        raise RuntimeError(f"anchor transaction reverted: {web3.to_hex(tx_hash)}")

    return {
        "network": NETWORK_NAME if transaction["chainId"] == CHAIN_ID else "custom",
        "tx_hash": web3.to_hex(tx_hash),
        "chain_id": transaction["chainId"],
        "block_number": receipt["blockNumber"],
        "from_address": account.address,
        "gas_used": receipt["gasUsed"],
    }
