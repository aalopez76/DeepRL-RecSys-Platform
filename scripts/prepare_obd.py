import argparse
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("prepare_obd")

def process_campaign(base_dir: Path, out_dir: Path, policy: str, campaign: str):
    csv_path = base_dir / policy / campaign / f"{campaign}.csv"
    item_context_path = base_dir / policy / campaign / "item_context.csv"
    
    if not csv_path.exists():
        logger.error(f"Falta el archivo principal de impresiones: {csv_path}")
        return
    if not item_context_path.exists():
        logger.error(f"Falta el archivo de contexto de items: {item_context_path}")
        return

    logger.info(f"Cargando item_context de {item_context_path}...")
    item_df = pd.read_csv(item_context_path)
    
    out_file = out_dir / policy / f"{campaign}.parquet"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    CHUNK_SIZE = 100000
    total_processed = 0
    writer = None
    
    for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE):
        merged = pd.merge(chunk, item_df, on="item_id", how="left")
        
        merged["user_item_affinity"] = merged.get("user-item_affinity_0", pd.Series([0.0]*len(merged))).fillna(0.0)
        merged["user_feature_0"] = merged.get("user_feature_0", pd.Series([""]*len(merged))).fillna("")
        
        for feat in ["item_feature_0", "item_feature_1", "item_feature_2", "item_feature_3"]:
            if feat in merged.columns:
                merged[feat] = merged[feat].fillna("")
                
        if "propensity_score" not in merged.columns or merged["propensity_score"].isnull().any():
            missing_probs = merged["propensity_score"].isnull().sum() if "propensity_score" in merged.columns else len(merged)
            logger.warning(f"Se encontraron {missing_probs} filas sin propensity, se rellenarán con 0.0")
            merged["propensity_score"] = merged.get("propensity_score", pd.Series([0.0]*len(merged))).fillna(0.0)
        
        def build_context(row):
            return json.dumps({
                "position": int(row.get("position", 1)),
                "user_features": str(row.get("user_feature_0", "")),
                "user_item_affinity": float(row.get("user_item_affinity", 0.0)),
                "item_features": [
                    str(row.get("item_feature_0", "")),
                    str(row.get("item_feature_1", "")),
                    str(row.get("item_feature_2", "")),
                    str(row.get("item_feature_3", ""))
                ]
            })
            
        merged["context"] = merged.apply(build_context, axis=1)
        
        mapped = pd.DataFrame()
        mapped["action"] = merged["item_id"].astype("int32")
        reward_col = "click" if "click" in merged.columns else "click_target_variable"
        if reward_col in merged.columns:
            mapped["reward"] = merged[reward_col].fillna(0).astype("int8")
        else:
            mapped["reward"] = pd.Series([0]*len(merged), dtype="int8")
            
        mapped["propensity"] = merged["propensity_score"].astype("float32")
        mapped["timestamp"] = pd.to_datetime(merged["timestamp"]).astype("datetime64[ns]")
        mapped["context"] = merged["context"].astype("string")
        
        # Guardar Chunk usando PyArrow
        table = pa.Table.from_pandas(mapped)
        if writer is None:
            writer = pq.ParquetWriter(out_file, table.schema, compression='snappy')
        writer.write_table(table)
        
        total_processed += len(mapped)
        logger.info(f"Procesados {total_processed} registros de {campaign}...")
        
    if writer:
        writer.close()
        
    logger.info(f"Total registros guardados en {out_file}: {total_processed}")

def main():
    parser = argparse.ArgumentParser(description="Prepare OBD dataset")
    parser.add_argument("--policy", type=str, default="random", help="Policy name (random, bts)")
    parser.add_argument("--campaign", type=str, default=None, help="Campaign name (all, men, women). If None, all will be processed.")
    parser.add_argument("--obd_dir", type=str, default=r"D:\Certificación\5.Proyecto\DeepRL-RecSys-Platform\open_bandit_dataset", help="Path to raw obd directory")
    parser.add_argument("--out_dir", type=str, default="data/obd", help="Path to output directory")
    args = parser.parse_args()

    base_dir = Path(args.obd_dir)
    out_dir = Path(args.out_dir)
    
    campaigns = [args.campaign] if args.campaign else ["all", "men", "women"]
    
    for c in campaigns:
        logger.info(f"=== Procesando campaña: {c} ===")
        try:
            process_campaign(base_dir, out_dir, args.policy, c)
        except Exception as e:
            logger.error(f"Error procesando la campaña {c}: {e}", exc_info=True)

if __name__ == "__main__":
    main()
