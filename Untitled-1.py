import json

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_technique_mapping(mitre_data):
    technique_mapping = {}
    
    # Map main techniques
    for tactic_idx, technique_ids in enumerate(mitre_data.get("Technique Id", [])):
        for idx, technique_id in enumerate(technique_ids):
            technique_name = mitre_data["Techniques"][tactic_idx][idx]
            technique_mapping[technique_id] = technique_name
    
    # Map sub-techniques if present
    if "Subtechnique" in mitre_data:
        for sub_list in mitre_data["Subtechnique"]:
            if isinstance(sub_list, list):  # Ensure we correctly process nested lists
                for sub in sub_list:
                    sub_id = sub.get("SubTechnique Id")
                    sub_name = sub.get("Sub Technique")
                    parent_technique = sub.get("Technique Name")  # Link to parent
                    
                    if sub_id and sub_name:
                        full_name = f"{parent_technique}: {sub_name}" if parent_technique else sub_name
                        technique_mapping[sub_id] = full_name
    
    return technique_mapping

def map_cve_to_techniques(cve_data, technique_mapping):
    mapped_data = []
    
    for entry in cve_data:
        cve_id = entry.get("CVE_ID", "Unknown")
        predictions = entry.get("Predictions", {})
        mapped_techniques = {}
        
        for technique_id, score in predictions.items():
            technique_id = f"T{technique_id}"
            technique_name = technique_mapping.get(technique_id, "Unknown Technique")
            mapped_techniques[technique_id] = {"name": technique_name, "score": score}
        
        mapped_data.append({"CVE_ID": cve_id, "Techniques": mapped_techniques})
    
    return mapped_data

def save_output(mapped_data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mapped_data, f, indent=4)

def main():
    cve_file = "output.json"
    mitre_file = "MITRE_ATT&CK_Tactics_Techniques.json"
    output_file = "mapped_output.json"
    
    cve_data = load_json(cve_file)
    mitre_data = load_json(mitre_file)
    
    technique_mapping = create_technique_mapping(mitre_data)
    mapped_data = map_cve_to_techniques(cve_data, technique_mapping)
    save_output(mapped_data, output_file)
    
    print(f"Mapping complete! Output saved to {output_file}")

if __name__ == "__main__":
    main()
