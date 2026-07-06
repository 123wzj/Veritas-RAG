from backend.evaluation.import_lexrag_kb import project_doc_id
from backend.evaluation.prepare_lexrag_dataset import (
    convert_dialogue,
    filter_valid_dialogues,
)
from backend.evaluation.run_lexrag_multiturn_eval import retrieval_metrics
from backend.services.ingestion.chunker import DocumentChunker


def test_filter_rejects_dialogue_with_missing_law():
    dialogues = [{
        "id": 1,
        "type": "[合同纠纷]",
        "conversation": [
            {"user": "问题", "assistant": "回答", "article": ["不存在法条"]}
        ] * 5,
    }]

    valid, rejected = filter_valid_dialogues(dialogues, {"存在法条"})

    assert not valid
    assert rejected[0]["missing_articles"] == ["不存在法条"]


def test_convert_preserves_five_turn_history_and_gold_context():
    source = {
        "id": 7,
        "type": "[合同纠纷]",
        "conversation": [{
            "user": f"问题{i}",
            "assistant": f"回答{i}",
            "article": ["法条A"],
            "article_context": [{"法条A": "证据正文"}],
            "keyword": ["合同"],
        } for i in range(5)],
    }

    converted = convert_dialogue(source)

    assert len(converted["turns"]) == 5
    assert converted["turns"][0]["requires_history"] is False
    assert converted["turns"][1]["requires_history"] is True
    assert converted["turns"][0]["gold_contexts"] == ["证据正文"]


def test_retrieval_metrics_map_law_name_to_project_doc_id():
    law_name = "《中华人民共和国民法典》第五百八十四条"
    mapped = project_doc_id(law_name)

    result = retrieval_metrics(
        [{"doc_id": mapped}],
        [law_name],
        [{"doc_id": mapped}],
        6,
    )

    assert result["gold_article_recall@6"] == 1.0
    assert result["gold_article_mrr@6"] == 1.0
    assert result["citation_to_gold_article"] == 1.0


def test_chunker_keeps_text_without_sentence_terminator():
    chunker = DocumentChunker()

    parents, children = chunker.chunk_document(
        {
            "type": "txt",
            "content": "《中华人民共和国刑法》第一百九十九条\n（删去）",
            "metadata": {"title": "《中华人民共和国刑法》第一百九十九条"},
        },
        "lex_deleted_article",
    )

    assert len(parents) == 1
    assert len(children) == 1
    assert children[0].content.endswith("（删去）")
