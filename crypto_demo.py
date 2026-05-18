import hashlib
import json

# Toy RSA values for demonstration only.
p = 61
q = 53
n = p * q
phi_n = (p - 1) * (q - 1)
e = 17
d = 2753

payload = {
    "device_id": "ret_bolus_0007",
    "timestamp": "2026-05-11T08:43:12Z",
    "sequence": 18452,
    "cow_id": "COW-0004",
    "body_temperature": 39.8,
    "rumination_minutes": 312,
    "activity_score": 41,
}


def canonicalize(data: dict) -> str:
    ordered_keys = [
        "device_id",
        "timestamp",
        "sequence",
        "cow_id",
        "body_temperature",
        "rumination_minutes",
        "activity_score",
    ]
    return "|".join(f"{key}={data[key]}" for key in ordered_keys)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rsa_sign(message_int: int) -> int:
    return pow(message_int, d, n)


def rsa_verify(signature_int: int) -> int:
    return pow(signature_int, e, n)


def main() -> None:
    canonical = canonicalize(payload)
    digest_hex = sha256_hex(canonical)
    digest_int = int(digest_hex, 16)
    reduced_digest = digest_int % n
    signature = rsa_sign(reduced_digest)
    verification = rsa_verify(signature)

    print("=== Toy RSA Signature Demonstration ===")
    print("Original Payload:")
    print(json.dumps(payload, indent=2))
    print("\nCanonical String:")
    print(canonical)
    print("\nSHA-256 Digest (hex):")
    print(digest_hex)
    print("\nDigest Reduced mod n:")
    print(reduced_digest)
    print("\nRSA Signature (s = m^d mod n):")
    print(signature)
    print("\nVerification (v = s^e mod n):")
    print(verification)
    print("\nSignature Valid:")
    print(verification == reduced_digest)

    tampered_payload = payload.copy()
    tampered_payload["body_temperature"] = 40.8
    tampered_canonical = canonicalize(tampered_payload)
    tampered_digest_hex = sha256_hex(tampered_canonical)
    tampered_reduced_digest = int(tampered_digest_hex, 16) % n

    print("\n=== Tamper Check ===")
    print("Tampered Canonical String:")
    print(tampered_canonical)
    print("\nTampered SHA-256 Digest (hex):")
    print(tampered_digest_hex)
    print("\nTampered Digest Reduced mod n:")
    print(tampered_reduced_digest)
    print("\nOld Signature Verified Against Tampered Payload:")
    print(verification == tampered_reduced_digest)


if __name__ == "__main__":
    main()
