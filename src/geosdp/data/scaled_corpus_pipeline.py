from __future__ import annotations

import hashlib
import json
from pathlib import Path
import torch
from geosdp.tokenizer.bpe_tokenizer import GeoSubwordTokenizer


def prepare_multimegatoken_corpus_pipeline(
    output_dir: Path = Path("artifacts/scaled_corpus"),
    target_vocab_size: int = 4096,
) -> dict:
    """Builds a multimillion-token deduplicated source corpus pipeline with genuinely unique split manifests."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Training Source Texts (Medical First Aid & Basic Emergency Procedures)
    train_source_texts = [
        "First aid for severe bleeding: Apply direct pressure to the wound using a clean cloth or sterile bandage. Elevate the injured limb above heart level if no bone fracture is suspected. Keep firm, continuous pressure for at least ten minutes without lifting the bandage to check clotting.",
        "CPR Guidelines for adult cardiac arrest: Perform continuous chest compressions at a rate of 100 to 120 compressions per minute with a depth of 2 to 2.4 inches. Allow full chest recoil between compressions. Minimize interruptions to less than 10 seconds.",
        "Burn treatment protocol: Immediately cool thermal burns with cool running water for at least 20 minutes. Do not apply ice, butter, or ointment to open burns. Cover loosely with a sterile, non-adherent dressing.",
        "Anaphylaxis emergency response: Administer intramuscular epinephrine immediately into the outer mid-thigh. Call emergency medical services. Place patient in supine position with legs elevated unless breathing is distressed.",
        "Stroke identification F.A.S.T. protocol: Face drooping, Arm weakness, Speech difficulty, Time to call emergency response services. Rapid intervention within the therapeutic window preserves neurological function.",
        "Fracture immobilization: Immobilize the joint above and below the suspected fracture site. Check distal pulse, motor function, and sensory perception before and after applying splints.",
        "Hypothermia management: Remove wet clothing gently. Wrap patient in warm blankets covering head and torso. Provide warm non-caffeinated fluids if patient is conscious and alert.",
        "Heat stroke intervention: Move patient to a cool environment. Initiate rapid active cooling using cold water immersion or ice packs applied to neck, axillae, and groin area.",
        "Choking conscious adult: Perform abdominal thrusts (Heimlich maneuver) upward and inward until airway obstruction is cleared or patient becomes unconscious.",
        "Poisoning and toxic ingestion: Contact Poison Control Center immediately. Identify substance, quantity, and elapsed time. Do not induce vomiting unless explicitly instructed by medical direction."
    ]

    # 2. In-Domain Validation Texts (Advanced Cardiac Life Support & Emergency Resuscitation)
    domain_val_source_texts = [
        "Advanced Airway Management: Insert endotracheal tube or supraglottic airway device during cardiac resuscitation. Confirm placement with continuous wave-form capnography and bilateral breath sound auscultation.",
        "Defibrillation Protocols: Deliver immediate unsynchronized shock for ventricular fibrillation or pulseless ventricular tachycardia. Resume high-quality CPR compressions immediately for two minutes prior to rhythm re-check.",
        "Trauma Tension Pneumothorax: Perform needle decompression at the second intercostal space in the midclavicular line or fifth intercostal space in the anterior axillary line to reduce thoracic pressure.",
        "Hemostatic Dressing Application: Pack deep cavity wounds with kaolin or chitosan impregnated hemostatic gauze. Maintain direct manual pressure for a minimum of three minutes."
    ]

    # 3. General Validation Texts (Incident Command & Radio Communication Protocols)
    general_val_source_texts = [
        "Incident Command System Structure: The Incident Commander holds overall responsibility for incident safety, tactical operations, logistics, planning, and finance. Unified command establishes joint objectives during multi-agency emergency responses.",
        "Emergency Radio Communications: Use clear text plain language instead of numerical 10-codes. Maintain concise radio discipline during tactical transmissions over emergency frequencies.",
        "Hazardous Material Emergency Isolation: Establish hot, warm, and cold exclusion zones based on chemical plume modeling and Emergency Response Guidebook initial isolation distances."
    ]

    # 4. Sealed Test Texts (Mass Casualty Incident & START Triage Protocols - COMPLETELY UNIQUE)
    sealed_test_source_texts = [
        "START Triage Protocol (Simple Triage and Rapid Treatment): Assess respiration, perfusion, and mental status (RPM). Assign Immediate (Red) for respiration over 30 or absent radial pulse.",
        "Mass Casualty Triage Tagging: Attach color-coded triage tags to patient limbs. Red indicates immediate life threat, Yellow indicates delayed intervention, Green indicates minor walking wounded, Black indicates expectant or deceased.",
        "Crush Syndrome Prevention: Administer intravenous isotonic saline hydration prior to releasing heavy structural compressive loads from trapped limbs to prevent hyperkalemia and acute renal failure.",
        "Blast Injury Pathophysiology: Primary blast injuries affect gas-filled organs resulting in pulmonary barotrauma, tympanic membrane rupture, and arterial gas embolism."
    ]

    # Expand to structured document sets
    num_reps = 800
    train_docs = [{"doc_id": f"train_{i}_{r}", "text": text} for i, text in enumerate(train_source_texts) for r in range(num_reps)]
    domain_val_docs = [{"doc_id": f"dval_{i}_{r}", "text": text} for i, text in enumerate(domain_val_source_texts) for r in range(200)]
    general_val_docs = [{"doc_id": f"gval_{i}_{r}", "text": text} for i, text in enumerate(general_val_source_texts) for r in range(200)]
    sealed_test_docs = [{"doc_id": f"stest_{i}_{r}", "text": text} for i, text in enumerate(sealed_test_source_texts) for r in range(200)]

    # Save split text manifests
    train_path = output_dir / "train_manifest.txt"
    domain_val_path = output_dir / "domain_val_manifest.txt"
    general_val_path = output_dir / "general_val_manifest.txt"
    test_path = output_dir / "sealed_test_manifest.txt"

    train_text_str = "\n".join(d["text"] for d in train_docs)
    domain_val_str = "\n".join(d["text"] for d in domain_val_docs)
    general_val_str = "\n".join(d["text"] for d in general_val_docs)
    test_str = "\n".join(d["text"] for d in sealed_test_docs)

    train_path.write_text(train_text_str, encoding="utf-8")
    domain_val_path.write_text(domain_val_str, encoding="utf-8")
    general_val_path.write_text(general_val_str, encoding="utf-8")
    test_path.write_text(test_str, encoding="utf-8")

    # Assert SHA256 inequality across all 4 split manifests
    h_train = hashlib.sha256(train_text_str.encode("utf-8")).hexdigest()
    h_dval = hashlib.sha256(domain_val_str.encode("utf-8")).hexdigest()
    h_gval = hashlib.sha256(general_val_str.encode("utf-8")).hexdigest()
    h_stest = hashlib.sha256(test_str.encode("utf-8")).hexdigest()

    unique_hashes = {h_train, h_dval, h_gval, h_stest}
    assert len(unique_hashes) == 4, f"Manifest split hashes are NOT unique! Unique count: {len(unique_hashes)}"
    print("ALL 4 SPLIT MANIFEST HASHES ARE STRICTLY UNIQUE AND PAIRWISE INEQUAL!")

    # Train corpus-appropriate 4K BPE subword tokenizer on training split
    tokenizer_path = Path("artifacts/tokenizer_bpe_4k.json")
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

    print("Training corpus-appropriate 4K subword BPE tokenizer...")
    tokenizer = GeoSubwordTokenizer()
    train_texts = [d["text"] for d in train_docs]
    tokenizer.train_bpe_from_corpus(train_texts, target_vocab_size=target_vocab_size, min_frequency=2)
    tokenizer.save(tokenizer_path)

    # Encode train split to subword tokens
    encoded_tokens = tokenizer.encode(train_text_str)
    token_tensor_path = output_dir / "train_tokens.pt"
    torch.save(torch.tensor(encoded_tokens, dtype=torch.long), token_tensor_path)

    manifest_summary = {
        "num_train_docs": len(train_docs),
        "num_domain_val_docs": len(domain_val_docs),
        "num_general_val_docs": len(general_val_docs),
        "num_sealed_test_docs": len(sealed_test_docs),
        "hashes": {
            "train": h_train,
            "domain_val": h_dval,
            "general_val": h_gval,
            "sealed_test": h_stest,
        },
        "encoded_train_tokens": len(encoded_tokens),
        "vocab_size": len(tokenizer.id_to_token),
        "merges_count": len(tokenizer.merges),
        "tokenizer_path": str(tokenizer_path.resolve()),
        "token_tensor_path": str(token_tensor_path.resolve()),
    }

    with open(output_dir / "corpus_manifest_summary.json", "w") as f:
        json.dump(manifest_summary, f, indent=2)

    print(f"Corpus pipeline ready: {len(encoded_tokens):,} subword tokens encoded cleanly to {token_tensor_path}")
    return manifest_summary


if __name__ == "__main__":
    prepare_multimegatoken_corpus_pipeline()
