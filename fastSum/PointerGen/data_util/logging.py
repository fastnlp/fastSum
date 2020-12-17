import logging
import logging.handlers

logger = logging.getLogger("ptr_gen_logger")

handler1 = logging.StreamHandler()

logger.setLevel(logging.DEBUG)
handler1.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
handler1.setFormatter(formatter)

logger.addHandler(handler1)
