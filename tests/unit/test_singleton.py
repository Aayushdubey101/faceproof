from face_engine.commons.logger import Logger

logger = Logger()


def test_singleton_same_object():
    assert Logger() == Logger()
    logger.info("✅ id's of instances of \"singletoned\" class Logger are the same")
