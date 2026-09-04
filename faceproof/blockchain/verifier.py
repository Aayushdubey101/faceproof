"""On-chain fingerprint verification.

Reads the anchored digest straight back out of the transaction calldata and
compares it with a locally recomputed fingerprint. Independent of the run that
created the evidence: nothing but the evidence file and the chain is needed.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from faceproof.blockchain.registry import connect


def read_anchor(tx_hash: str, rpc_url: Optional[str] = None) -> str:
    """Read the digest out of the on-chain transaction's calldata."""
    from web3.exceptions import Web3Exception  # deferred, as in registry.connect

    web3 = connect(rpc_url)
    try:
        transaction = web3.eth.get_transaction(tx_hash)
    except (Web3Exception, ValueError) as error:
        raise RuntimeError(f"cannot read anchor transaction {tx_hash}: {error}") from None

    calldata: Any = transaction["input"]
    if hasattr(calldata, "hex"):
        calldata = calldata.hex()
    return str(calldata).removeprefix("0x").lower()


def verify_onchain(
    digest_hex: str,
    tx_hash: str,
    rpc_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare a locally recomputed fingerprint against the on-chain record."""
    onchain = read_anchor(tx_hash, rpc_url)
    return {
        "tx_hash": tx_hash,
        "onchain_digest": onchain,
        "local_digest": digest_hex.lower(),
        "matches": onchain == digest_hex.lower(),
    }
