# RSA demo for the cattle-health project

## Important note
This demo has **two layers**:

1. **Practical project logic**: the device hashes the payload with SHA-256 and signs it with RSA so the server can verify origin and integrity.
2. **Toy arithmetic RSA example**: small numbers are used only so the math can be shown visibly. Real RSA uses much larger keys and padded signature schemes.

## 1) Sample project payload

```json
{
  "device_id": "ret_bolus_0007",
  "timestamp": "2026-05-11T08:43:12Z",
  "sequence": 18452,
  "cow_id": "COW-0004",
  "body_temperature": 39.8,
  "rumination_minutes": 312,
  "activity_score": 41
}
```

## 2) Canonical string used for hashing

```text
device_id=ret_bolus_0007|timestamp=2026-05-11T08:43:12Z|sequence=18452|cow_id=COW-0004|body_temperature=39.8|rumination_minutes=312|activity_score=41
```

## 3) SHA-256 hash of the canonical string

```text
54081c6afcfe2c0ed193ebb75824887f3b320533e63658a4c470b918d19f47c2
```

This is the exact 32-byte digest in hexadecimal form.

## 4) Practical project workflow

### Device side
- Build canonical payload
- Compute SHA-256 digest
- Sign the digest with the device's RSA private key
- Send payload + signature

### Server side
- Rebuild the canonical string from the received payload
- Recompute SHA-256 digest
- Verify the RSA signature with the stored public key
- Check timestamp and sequence freshness
- Accept or reject the record

## 5) Toy RSA arithmetic demo (for defense/explanation)

### Toy RSA parameters
- p = 61
- q = 53
- n = p × q = 3233
- phi(n) = (p - 1)(q - 1) = 3120
- public exponent e = 17
- private exponent d = 2753

### Convert the hash to a small demo integer
Real digests are much larger than the toy modulus n, so for the arithmetic demo:

```text
m = int(SHA256_digest, 16) mod n
m = 2180
```

### Signing step
Toy signature:

```text
s = m^d mod n
s = 2180^2753 mod 3233
s = 2932
```

### Verification step
The server verifies with the public exponent:

```text
m_check = s^e mod n
m_check = 2932^17 mod 3233
m_check = 2180
```

Because:

```text
m_check = m = 2180
```

the toy signature verifies correctly.

## 6) Optional toy RSA encryption example
This project mainly uses RSA for **signatures**, not for encrypting every payload.  
But if you want to visually show how RSA encryption transforms a small number, take:

```text
session_fragment = 123
ciphertext = 123^17 mod 3233 = 855
plaintext = 855^2753 mod 3233 = 123
```

So the value changes:

```text
123  ->  855  ->  123
```

This is only a tiny arithmetic demo. Real systems usually use RSA to protect a key or to sign data, not to encrypt large sensor payloads directly.

## 7) What each transformation adds

- **Canonicalization**: ensures both sides hash exactly the same content.
- **SHA-256**: compresses the payload into a fixed-length digest that changes completely if the payload changes.
- **RSA signature**: proves the digest came from the holder of the private key.
- **Verification**: confirms origin and integrity before the record enters prediction.
- **Timestamp + sequence**: helps detect replayed old packets.

## 8) Submission-ready short interpretation

In the project, the sensor or gateway does not send “trusted” data by default.  
It sends a payload that is transformed into a canonical string, hashed with SHA-256, and signed with RSA.  
The backend reconstructs the same string, recomputes the hash, and verifies the RSA signature with the registered public key.  
If the signature matches and the timestamp/sequence checks pass, the observation is accepted; otherwise it is rejected and logged as a security event.
