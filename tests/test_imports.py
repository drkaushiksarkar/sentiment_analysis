def test_package_imports() -> None:
    import sentiment_package  # noqa: F401
    from sentiment_package import imdb, sarcasm  # noqa: F401

    assert isinstance(sentiment_package.__version__, str)
