from frauddet.app import create_app


def test_index_renders():
    client = create_app().test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"Card fraud detector" in response.data


def test_score_sample():
    client = create_app().test_client()
    response = client.post("/score-sample", follow_redirects=True)
    assert response.status_code == 200
    assert b"Prediction" in response.data or b"Train a model" in response.data
