import argparse
from abc import ABC
from abc import abstractmethod
from typing import List
import time
import asyncio
import numpy as np
import soundfile
import requests
import json

import threading
import os
import yaml
import hashlib
import uuid


import functools
import logging
import inspect
import websockets

import paddle
import paddleaudio

__all__ = [
    "logger",
]

try:
    from .. import __version__
except ImportError:
    __version__ = "0.0.0"  # for develop branch


class TextHttpHandler:
    def __init__(self, server_ip="127.0.0.1", port=8090):
        """Text http client request

        Args:
            server_ip (str, optional): the text server ip. Defaults to "127.0.0.1".
            port (int, optional): the text server port. Defaults to 8090.
        """
        super().__init__()
        self.server_ip = server_ip
        self.port = port
        if server_ip is None or port is None:
            self.url = None
        else:
            self.url = "http://" + self.server_ip + ":" + str(self.port) + "/paddlespeech/text"
        # logger.info(f"endpoint: {self.url}")

    def run(self, text):
        """Call the text server to process the specific text

        Args:
            text (str): the text to be processed

        Returns:
            str: punctuation text
        """
        if self.server_ip is None or self.port is None:
            return text
        request = {
            "text": text,
        }
        try:
            res = requests.post(url=self.url, data=json.dumps(request))
            response_dict = res.json()
            punc_text = response_dict["result"]["punc_text"]
        except Exception as e:
            logger.error(f"Call punctuation {self.url} occurs error")
            logger.error(e)
            punc_text = text

        return punc_text


class ASRWsAudioHandler:
    def __init__(self, url=None, port=None, endpoint="/paddlespeech/asr/streaming", punc_server_ip=None, punc_server_port=None):
        """PaddleSpeech Online ASR Server Client  audio handler
           Online asr server use the websocket protocal
        Args:
            url (str, optional): the server ip. Defaults to None.
            port (int, optional): the server port. Defaults to None.
            endpoint(str, optional): to compatiable with python server and c++ server.
            punc_server_ip(str, optional): the punctuation server ip. Defaults to None.
            punc_server_port(int, optional): the punctuation port. Defaults to None
        """
        self.url = url
        self.port = port
        if url is None or port is None or endpoint is None:
            self.url = None
        else:
            self.url = "ws://" + self.url + ":" + str(self.port) + endpoint
        self.punc_server = TextHttpHandler(punc_server_ip, punc_server_port)
        # logger.info(f"endpoint: {self.url}")

    def read_wave(self, wavfile_path: str):
        """read the audio file from specific wavfile path

        Args:
            wavfile_path (str): the audio wavfile,
                                 we assume that audio sample rate matches the model

        Yields:
            numpy.array: the samall package audio pcm data
        """
        samples, sample_rate = soundfile.read(wavfile_path, dtype="int16")
        x_len = len(samples)
        assert sample_rate == 16000

        chunk_size = int(85 * sample_rate / 1000)  # 85ms, sample_rate = 16kHz

        if x_len % chunk_size != 0:
            padding_len_x = chunk_size - x_len % chunk_size
        else:
            padding_len_x = 0

        padding = np.zeros((padding_len_x), dtype=samples.dtype)
        padded_x = np.concatenate([samples, padding], axis=0)

        assert (x_len + padding_len_x) % chunk_size == 0
        num_chunk = (x_len + padding_len_x) / chunk_size
        num_chunk = int(num_chunk)
        for i in range(0, num_chunk):
            start = i * chunk_size
            end = start + chunk_size
            x_chunk = padded_x[start:end]
            yield x_chunk

    async def run(self, wavfile_path: str):
        """Send a audio file to online server

        Args:
            wavfile_path (str): audio path

        Returns:
            str: the final asr result
        """
        logging.debug("send a message to the server")

        if self.url is None:
            logger.error("No asr server, please input valid ip and port")
            return ""

        # 1. send websocket handshake protocal
        start_time = time.time()
        async with websockets.connect(self.url) as ws:
            # 2. server has already received handshake protocal
            # client start to send the command
            audio_info = json.dumps({"name": "test.wav", "signal": "start", "nbest": 1}, sort_keys=True, indent=4, separators=(",", ": "))
            await ws.send(audio_info)
            msg = await ws.recv()
            # logger.info("client receive msg={}".format(msg))

            # 3. send chunk audio data to engine
            for chunk_data in self.read_wave(wavfile_path):
                await ws.send(chunk_data.tobytes())
                msg = await ws.recv()
                msg = json.loads(msg)
                # logger.info("client receive msg={}".format(msg))
            # client start to punctuation restore
            if self.punc_server and len(msg["result"]) > 0:
                msg["result"] = self.punc_server.run(msg["result"])
                # logger.info("client punctuation restored msg={}".format(msg))
            # 4. we must send finished signal to the server
            audio_info = json.dumps({"name": "test.wav", "signal": "end", "nbest": 1}, sort_keys=True, indent=4, separators=(",", ": "))
            await ws.send(audio_info)
            msg = await ws.recv()

            # 5. decode the bytes to str
            msg = json.loads(msg)

            if self.punc_server:
                msg["result"] = self.punc_server.run(msg["result"])

            # 6. logging the final result and comptute the statstics
            elapsed_time = time.time() - start_time
            audio_info = soundfile.info(wavfile_path)
            # logger.info("client final receive msg={}".format(msg))
            # logger.info(f"audio duration: {audio_info.duration}, elapsed time: {elapsed_time}, RTF={elapsed_time/audio_info.duration}")

            result = msg

            return result


def stats_wrapper(executor_func):
    def _warpper(self, *args, **kwargs):
        try:
            _note_one_stat(type(self).__name__, _parse_args(executor_func, *args, **kwargs))
        except Exception:
            pass
        return executor_func(self, *args, **kwargs)

    return _warpper


def _parse_args(func, *args, **kwargs):
    # FullArgSpec(args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations)
    argspec = inspect.getfullargspec(func)

    keys = argspec[0]
    if keys[0] == "self":  # Remove self pointer.
        keys = keys[1:]

    default_values = argspec[3]
    values = [None] * (len(keys) - len(default_values))
    values.extend(list(default_values))
    params = dict(zip(keys, values))

    for idx, v in enumerate(args):
        params[keys[idx]] = v
    for k, v in kwargs.items():
        params[k] = v

    return params


def _note_one_stat(cls_name, params={}):
    task = cls_name.replace("Executor", "").lower()  # XXExecutor
    extra_info = {
        "paddle_version": paddle.__version__,
    }

    if "model" in params:
        model = params["model"]
    else:
        model = None

    if "audio_file" in params:
        try:
            _, sr = paddleaudio.backends.soundfile_load(params["audio_file"])
        except Exception:
            sr = -1

    if task == "asr":
        extra_info.update(
            {
                "lang": params["lang"],
                "inp_sr": sr,
                "model_sr": params["sample_rate"],
            }
        )
    elif task == "st":
        extra_info.update(
            {
                "lang": params["src_lang"] + "-" + params["tgt_lang"],
                "inp_sr": sr,
                "model_sr": params["sample_rate"],
            }
        )
    elif task == "tts":
        model = params["am"]
        extra_info.update(
            {
                "lang": params["lang"],
                "vocoder": params["voc"],
            }
        )
    elif task == "cls":
        extra_info.update(
            {
                "inp_sr": sr,
            }
        )
    elif task == "text":
        extra_info.update(
            {
                "sub_task": params["task"],
                "lang": params["lang"],
            }
        )
    else:
        return

    StatsWorker(
        task=task,
        model=model,
        version=__version__,
        extra_info=extra_info,
    ).start()


def _get_user_home():
    return os.path.expanduser("~")


def _get_paddlespcceh_home():
    if "PPSPEECH_HOME" in os.environ:
        home_path = os.environ["PPSPEECH_HOME"]
        if os.path.exists(home_path):
            if os.path.isdir(home_path):
                return home_path
            else:
                raise RuntimeError("The environment variable PPSPEECH_HOME {} is not a directory.".format(home_path))
        else:
            return home_path
    return os.path.join(_get_user_home(), ".paddlespeech")


def _get_sub_home(directory):
    home = os.path.join(_get_paddlespcceh_home(), directory)
    if not os.path.exists(home):
        os.makedirs(home)
    return home


PPSPEECH_HOME = _get_paddlespcceh_home()
MODEL_HOME = _get_sub_home("models")
CONF_HOME = _get_sub_home("conf")


def _md5(text: str):
    """Calculate the md5 value of the input text."""
    md5code = hashlib.md5(text.encode())
    return md5code.hexdigest()


class ConfigCache:
    def __init__(self):
        self._data = {}
        self._initialize()
        self.file = os.path.join(CONF_HOME, "cache.yaml")
        if not os.path.exists(self.file):
            self.flush()
            return

        with open(self.file, "r") as file:
            try:
                cfg = yaml.load(file, Loader=yaml.FullLoader)
                self._data.update(cfg)
            except BaseException:
                self.flush()

    @property
    def cache_info(self):
        return self._data["cache_info"]

    def _initialize(self):
        # Set default configuration values.
        cache_info = _md5(str(uuid.uuid1())[-12:]) + "-" + str(int(time.time()))
        self._data["cache_info"] = cache_info

    def flush(self):
        """Flush the current configuration into the configuration file."""
        with open(self.file, "w") as file:
            cfg = json.loads(json.dumps(self._data))
            yaml.dump(cfg, file)


stats_api = "http://paddlepaddle.org.cn/paddlehub/stat"
cache_info = ConfigCache().cache_info


class StatsWorker(threading.Thread):
    def __init__(self, task="asr", model=None, version=__version__, extra_info={}):
        threading.Thread.__init__(self)
        self._task = task
        self._model = model
        self._version = version
        self._extra_info = extra_info

    def run(self):
        params = {"task": self._task, "version": self._version, "from": "ppspeech"}
        if self._model:
            params["model"] = self._model

        self._extra_info.update(
            {
                "cache_info": cache_info,
            }
        )
        params.update({"extra": json.dumps(self._extra_info)})

        try:
            requests.get(stats_api, params)
        except Exception:
            pass

        return


class Logger(object):
    def __init__(self, name: str = None):
        name = "PaddleSpeech" if not name else name
        self.logger = logging.getLogger(name)

        log_config = {
            "DEBUG": 10,
            "INFO": 20,
            "TRAIN": 21,
            "EVAL": 22,
            "WARNING": 30,
            "ERROR": 40,
            "CRITICAL": 50,
            "EXCEPTION": 100,
        }
        for key, level in log_config.items():
            logging.addLevelName(level, key)
            if key == "EXCEPTION":
                self.__dict__[key.lower()] = self.logger.exception
            else:
                self.__dict__[key.lower()] = functools.partial(self.__call__, level)

        self.format = logging.Formatter(fmt="[%(asctime)-15s] [%(levelname)8s] - %(message)s")

        self.handler = logging.StreamHandler()
        self.handler.setFormatter(self.format)

        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

    def __call__(self, log_level: str, msg: str):
        self.logger.log(log_level, msg)


logger = Logger()


class BaseExecutor(ABC):
    """
    An abstract executor of paddlespeech server tasks.
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser()

    @abstractmethod
    def execute(self, argv: List[str]) -> bool:
        """
        Command line entry. This method can only be accessed by a command line such as `paddlespeech asr`.

        Args:
            argv (List[str]): Arguments from command line.

        Returns:
            int: Result of the command execution. `True` for a success and `False` for a failure.
        """
        pass

    @abstractmethod
    def __call__(self, *arg, **kwargs):
        """
        Python API to call an executor.
        """
        pass


class ASROnlineClientExecutor(BaseExecutor):
    def __init__(self):
        super(ASROnlineClientExecutor, self).__init__()
        self.parser = argparse.ArgumentParser(prog="paddlespeech_client.asr_online", add_help=True)
        self.parser.add_argument("--server_ip", type=str, default="127.0.0.1", help="server ip")
        self.parser.add_argument("--port", type=int, default=8091, help="server port")
        self.parser.add_argument("--input", type=str, default=None, help="Audio file to be recognized", required=True)
        self.parser.add_argument("--sample_rate", type=int, default=16000, help="audio sample rate")
        self.parser.add_argument("--lang", type=str, default="zh_cn", help="language")
        self.parser.add_argument("--audio_format", type=str, default="wav", help="audio format")
        self.parser.add_argument("--punc.server_ip", type=str, default=None, dest="punc_server_ip", help="Punctuation server ip")
        self.parser.add_argument("--punc.port", type=int, default=8190, dest="punc_server_port", help="Punctuation server port")

    def execute(self, argv: List[str]) -> bool:
        args = self.parser.parse_args(argv)
        input_ = args.input
        server_ip = args.server_ip
        port = args.port
        sample_rate = args.sample_rate
        lang = args.lang
        audio_format = args.audio_format
        try:
            time_start = time.time()
            res = self(
                input=input_, server_ip=server_ip, port=port, sample_rate=sample_rate, lang=lang, audio_format=audio_format, punc_server_ip=args.punc_server_ip, punc_server_port=args.punc_server_port
            )
            time_end = time.time()
            # logger.info(res)
            # logger.info("Response time %f s." % (time_end - time_start))
            return True
        except Exception as e:
            # logger.error("Failed to speech recognition.")
            # logger.error(e)
            return False

    @stats_wrapper
    def __call__(
        self,
        input: str,
        server_ip: str = "127.0.0.1",
        port: int = 8091,
        sample_rate: int = 16000,
        lang: str = "zh_cn",
        audio_format: str = "wav",
        punc_server_ip: str = None,
        punc_server_port: str = None,
    ):
        """Python API to call asr online executor.

        Args:
            input (str): the audio file to be send to streaming asr service.
            server_ip (str, optional): streaming asr server ip. Defaults to "127.0.0.1".
            port (int, optional): streaming asr server port. Defaults to 8091.
            sample_rate (int, optional): audio sample rate. Defaults to 16000.
            lang (str, optional): audio language type. Defaults to "zh_cn".
            audio_format (str, optional): audio format. Defaults to "wav".
            punc_server_ip (str, optional): punctuation server ip. Defaults to None.
            punc_server_port (str, optional): punctuation server port. Defaults to None.

        Returns:
            str: the audio text
        """

        # logger.info("asr websocket client start")
        handler = ASRWsAudioHandler(server_ip, port, punc_server_ip=punc_server_ip, punc_server_port=punc_server_port)
        loop = asyncio.get_event_loop()
        res = loop.run_until_complete(handler.run(input))
        # logger.info("asr websocket client finished")

        return res["result"]
