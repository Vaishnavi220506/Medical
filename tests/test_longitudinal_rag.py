from oncovision.analysis.longitudinal import response_category
from oncovision.llm.rag import knowledge_base


def _s(product, diameter=10.0):
    return {"measurements": {"bidimensional_product": product, "longest_diameter": diameter}}


def test_rano_categories():
    assert response_category("brain", _s(100), _s(140))["category"] == "progressive disease"
    assert response_category("brain", _s(100), _s(40))["category"] == "partial response"
    assert response_category("brain", _s(100), _s(110))["category"] == "stable disease"
    assert response_category("brain", _s(100), _s(0))["category"] == "complete response"
    assert response_category("brain", _s(0), _s(50))["category"] == "new lesion"


def test_recist_categories():
    assert response_category("breast", _s(0, 10), _s(0, 12.5))["category"] == "progressive disease"
    assert response_category("breast", _s(0, 10), _s(0, 6.5))["category"] == "partial response"


def test_knowledge_search():
    kb = knowledge_base()
    top = kb.search("dural tail extra-axial enhancement", k=1)[0][0]
    assert top.title == "Meningioma"
    top = kb.search("BI-RADS category 4 biopsy", k=1)[0][0]
    assert "Breast" in top.title
    assert all(c.url.startswith("https://") for c in kb.chunks)
