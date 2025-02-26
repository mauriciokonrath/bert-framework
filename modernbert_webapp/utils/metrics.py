from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

def calculate_metrics(full_text_result, chunked_results):
    """
    Calcula métricas a partir das previsões do modelo.
    """
    try:
        # Obtém a probabilidade da classe 1 (sensível)
        full_text_prob_class_1 = full_text_result.get("probabilities", [0, 0])[1]  # Índice 1 = probabilidade de classe 1

        # Obtém previsões e labels reais para chunks
        chunk_preds = [chunk["prediction"] for chunk in chunked_results]
        chunk_labels = [1] * len(chunk_preds)  # Supondo que todos sejam da classe 1 (ajuste se necessário)
        print("chunk_preds: ", chunk_preds)
        print("chunk_labels: ", chunk_labels)
        # Calcula métricas
        accuracy = accuracy_score(chunk_labels, chunk_preds)
        precision = precision_score(chunk_labels, chunk_preds, zero_division=1)
        recall = recall_score(chunk_labels, chunk_preds, zero_division=1)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        classification_rep = classification_report(chunk_labels, chunk_preds, output_dict=True, zero_division=1)


        # Calcula a média das probabilidades de classe 1 nos chunks
        avg_prob_class_1 = sum(chunk.get("probabilities", [0, 0])[1] for chunk in chunked_results) / len(chunked_results)

        return {
            "full_text_class_1": full_text_prob_class_1,
            "average_chunk_class_1": avg_prob_class_1,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "classification_report": classification_rep
        }
    except Exception as e:
        print(f"❌ Erro ao calcular métricas: {e}")
        return {"error": f"Erro ao calcular métricas: {e}"}
