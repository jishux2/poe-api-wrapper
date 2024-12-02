import ballyregan

try:
    from ballyregan.models import Protocols

    PROXY = True
except:
    PROXY = False

if PROXY:

    def fetch_proxy():
        http = ballyregan.Proxy(
            ip="127.0.0.1",
            port="10809",
            protocol=Protocols.HTTP,
        )
        https = ballyregan.Proxy(
            ip="127.0.0.1",
            port="10809",
            protocol=Protocols.HTTPS,
        )
        return [http, https]
