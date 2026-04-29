import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. FILE CONFIGURATION
# ==========================================
# The file you generated at the end of your pipeline
FILE_MIMIC = "" 

# The file with vectorized LOINC codes (which you prepared by passing LOINC through SapBERT)
FILE_LOINC_TARGET = "loinc_sapbert_embeddings.pkl" 

# The original MIMIC table for Ground Truth
FILE_D_LABITEMS = "d_labitems.csv" 

def run_benchmark():
    print("⏳ Loading spatial data...")
    df_mimic = pd.read_pickle(FILE_MIMIC)
    df_loinc = pd.read_pickle(FILE_LOINC_TARGET)
    df_ground_truth = pd.read_csv(FILE_D_LABITEMS, usecols=['itemid', 'label'])

    print("🚀 Extracting vector matrices...")
    # Transform lists of numbers into Numpy matrices for ultra-fast computation
    matrix_mimic = np.vstack(df_mimic['embedding_vector'].values)
    matrix_loinc = np.vstack(df_loinc['embedding_vector'].values)

    print("🧮 Calculating Cosine Similarity...")
    # This single line calculates distances for ALL vs ALL
    similarity = cosine_similarity(matrix_mimic, matrix_loinc)

    # ==========================================
    # 2. EXTRACTION OF TOP-5 PREDICTIONS
    # ==========================================
    print("🏆 Extracting Top-1, Top-3 and Top-5...")
    
    # argsort sorts from smallest to largest. [:, ::-1] reverses to have the largest (most similar) first
    sorted_indices = np.argsort(similarity, axis=1)[:, ::-1]

    results = []

    for i, row in df_mimic.iterrows():
        current_item_id = row['itemid'] # or 'ground_truth_itemid' depending on how you named it
        generated_string = row['semantic_string_clean']
        
        # Get the indices of the 5 most similar LOINC codes for this row
        top_5_indices = sorted_indices[i, :5]
        
        # Retrieve actual LOINC data using these indices
        top_5_loinc = df_loinc.iloc[top_5_indices]
        
        predictions = {
            "itemid": current_item_id,
            "llm_string": generated_string,
            "Top1_LOINC": top_5_loinc.iloc[0]['LOINC_NUM'],
            "Top1_Name": top_5_loinc.iloc[0]['LOINC_NAME'],
            "Top1_Score": similarity[i, top_5_indices[0]],
            
            "Top2_LOINC": top_5_loinc.iloc[1]['LOINC_NUM'],
            "Top2_Name": top_5_loinc.iloc[1]['LOINC_NAME'],
            
            "Top3_LOINC": top_5_loinc.iloc[2]['LOINC_NUM'],
            "Top3_Name": top_5_loinc.iloc[2]['LOINC_NAME'],
            
            "Top4_LOINC": top_5_loinc.iloc[3]['LOINC_NUM'],
            "Top4_Name": top_5_loinc.iloc[3]['LOINC_NAME'],
            
            "Top5_LOINC": top_5_loinc.iloc[4]['LOINC_NUM'],
            "Top5_Name": top_5_loinc.iloc[4]['LOINC_NAME'],
        }
        results.append(predictions)

    df_results = pd.DataFrame(results)

    # ==========================================
    # 3. MERGE WITH GROUND TRUTH AND SAVE
    # ==========================================
    print("🔗 Merging with Ground Truth (d_labitems)...")
    df_final = pd.merge(df_results, df_ground_truth, on='itemid', how='left')
    
    # Rename the MIMIC label column to highlight that it's our Ground Truth
    df_final.rename(columns={'label': 'MIMIC_Ground_Truth_Label'}, inplace=True)
    
    # Reorder columns for perfect reading in Excel
    final_columns = ['itemid', 'MIMIC_Ground_Truth_Label', 'llm_string', 
                      'Top1_Score', 'Top1_LOINC', 'Top1_Name', 
                      'Top2_LOINC', 'Top2_Name', 'Top3_LOINC', 'Top3_Name', 
                      'Top4_LOINC', 'Top4_Name', 'Top5_LOINC', 'Top5_Name']
    
    df_final = df_final[final_columns]

    df_final.to_csv("report_benchmark_final.csv", index=False)
    print("\n✅ DONE! Results saved to 'report_benchmark_final.csv'")

if __name__ == "__main__":
    run_benchmark()