from httpx import Client
from bs4 import BeautifulSoup
from loguru import logger
import quickjs
import re

class PoeBundle:
    form_key_pattern = r"window\.([a-zA-Z0-9]+)=function\(\)\{return window"
    window_secret_pattern = r'let useFormkeyDecode=[\s\S]*?(window\.[\w]+="[^"]+")'
    static_pattern = r'static[^"]*\.js'
    revision_pattern = r'"poe-revision":\s*"([a-f0-9]{40})"'

    def __init__(self, document: str):
        self._window = "const window={document:{hack:1},navigator:{userAgent:'safari <3'}};"
        self._src_scripts = []
        self._webpack_script: str = None
        self._revision: str = None

        self.init_window(document)

    def init_window(self, document: str):
        # initialize the window object with document scripts
        logger.info("Initializing web data")

        scripts = BeautifulSoup(document, "html.parser").find_all('script')
        for script in scripts:
            if (src := script.attrs.get("src")) and (src not in self._src_scripts):
                if "_app" in src:
                    self.init_app(src)
                if "buildManifest" in src:
                    self.extend_src_scripts(src)
                elif "webpack" in src:
                    self._webpack_script = src
                    self.extend_src_scripts(src)
                else:
                    self._src_scripts.append(src)
            elif ("document." in script.text) or ("function" not in script.text):
                continue
            elif script.attrs.get("type") == "application/json":
                continue
            self._window += script.text

        logger.info("Web data initialized")

    def init_app(self, src: str):
        script = self.load_src_script(src)
        
        # 获取window secret
        if not (window_secret_match := re.search(self.window_secret_pattern, script)):
            raise RuntimeError("Failed to find window secret in js scripts")
        self._window += window_secret_match.group(1) + ';'
        
        # 获取revision
        if not (revision_match := re.search(self.revision_pattern, script)):
            raise RuntimeError("Failed to find poe-revision in app script")
        self._revision = revision_match.group(1)
        logger.info(f"Retrieved poe-revision successfully: {self._revision}")

    def extend_src_scripts(self, manifest_src: str):
        # extend src scripts list with static scripts from manifest
        static_main_url = self.get_base_url(manifest_src)
        manifest = self.load_src_script(manifest_src)

        matches = re.findall(self.static_pattern, manifest)
        scr_list = [f"{static_main_url}{match}" for match in matches]

        self._src_scripts.extend(scr_list)

    def get_revision(self) -> str:
        if not self._revision:
            raise RuntimeError("Revision not initialized")
        return self._revision

    @staticmethod
    def load_src_script(src: str) -> str:
        with Client() as client:
            resp = client.get(src)
        if resp.status_code != 200:
            logger.warning(f"Failed to load script {src}, status code: {resp.status_code}")
        return resp.text

    @staticmethod
    def get_base_url(src: str) -> str:
        return src.split("static/")[0]

    def get_form_key(self) -> str:
        script = self._window

        match = re.search(self.form_key_pattern, script)
        if not (secret := match.group(1)):
            raise RuntimeError("Failed to parse form-key function in Poe document")
        
        script += f'window.{secret}().slice(0, 32);'
        context = quickjs.Context()
        formkey = str(context.eval(script))
        logger.info(f"Retrieved formkey successfully: {formkey}")
        return formkey
