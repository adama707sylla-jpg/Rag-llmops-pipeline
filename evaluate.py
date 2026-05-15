"""
evaluate.py — LLM Evaluation avec Ragas
Semaine 4 : RAG LLMOps Pipeline

Métriques :
  - faithfulness       : la réponse est-elle fidèle au contexte ?
  - answer_relevancy   : la réponse répond-elle à la question ?
  - context_recall     : le bon contexte est-il récupéré ?

Retourne exit code 1 si un seuil est dépassé (CI fail).
"""

import json
import sys
import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from ragas.run_config import RunConfig

#  LLM + Embeddings locaux (Ollama) 
OLLAMA_URL = "http://localhost:11434"

llm = LangchainLLMWrapper(Ollama(
    model="phi3:mini",
    base_url=OLLAMA_URL,
    timeout=300,
))

embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(
    model="phi3:mini",
    base_url=OLLAMA_URL,
))
# Seuils CI 
THRESHOLDS = {
    "faithfulness":     0.5,
    "answer_relevancy": 0.7,
    "context_recall":   0.7,
}

DATASET_PATH = os.path.join(os.path.dirname(__file__), "eval_dataset.json")


def load_dataset(path: str) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return Dataset.from_dict({
        "question":     [d["question"]     for d in data],
        "answer":       [d["answer"]       for d in data],
        "contexts":     [d["contexts"]     for d in data],
        "ground_truth": [d["ground_truth"] for d in data],
    })

  # Force séquentiel pour Ollama local
    import ragas
    ragas.evaluate.__globals__['executor_factory'] = None

def run_evaluation(dataset: Dataset) -> pd.DataFrame:
    run_config = RunConfig(
        timeout=600,
        max_workers=1,   # séquentiel — un appel à la fois
        max_retries=1,
    )
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
        run_config=run_config,
    )
    return result.to_pandas()

def check_thresholds(df: pd.DataFrame) -> bool:
    """Retourne True si tous les seuils sont respectés."""
    
    passed = True
    print("\n─── Résultats Ragas ───────────────────────────────────────")
    for metric, threshold in THRESHOLDS.items():
        if metric not in df.columns:
            print(f"  ⚠️  Métrique '{metric}' absente des résultats")
            continue
        score = df[metric].mean()
        status = "✅" if score >= threshold else "❌"
        print(f"  {status}  {metric:<22} score={score:.3f}  seuil={threshold}")
        if score < threshold:
            passed = False
    print("───────────────────────────────────────────────────────────\n")
    return passed


def main():
    print("🔍 Chargement du dataset d'évaluation...")
    dataset = load_dataset(DATASET_PATH)
    print(f"   {len(dataset)} exemples chargés\n")

    print("⚙️  Lancement de l'évaluation Ragas...")
    df = run_evaluation(dataset)

    # Sauvegarde CSV pour traçabilité
    out_csv = "eval_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"📄 Résultats sauvegardés dans {out_csv}")

    passed = check_thresholds(df)

    if passed:
        print("🎉 Tous les seuils sont respectés — CI PASS")
        sys.exit(0)
    else:
        print("💥 Un ou plusieurs seuils non atteints — CI FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()