import os
import queue
import re
import socket
import sys
import time
from typing import Optional
from threading import Thread

server: str = "irc.chat.twitch.tv"
port: int = 6667
nickname: str = "kscalelabs"
token: str = os.environ.get("STOMPYLIVE_TOKEN")
channel: str = "#kscalelabs"

message_queue: queue.Queue[str] = queue.Queue()

def parse_message(raw_message: str) -> Optional[str]:
    # This regex pattern matches the message part of a PRIVMSG
    pattern: str = r"^:.+!.+@.+\.tmi\.twitch\.tv PRIVMSG #\w+ :(.+)$"
    match = re.match(pattern, raw_message)
    if match:
        return match.group(1)
    return None

def init() -> None:
    print("Connecting to Twitch IRC")
    sock: socket.socket = socket.socket()

    try:
        sock.connect((server, port))
        print("Connected to server")
        
        sock.send(f"PASS {token}\n".encode("utf-8"))
        sock.send(f"NICK {nickname}\n".encode("utf-8"))
        sock.send(f"JOIN {channel}\n".encode("utf-8"))

        last_ping: float = time.time()

        while True:
            if time.time() - last_ping > 240:
                sock.send("PING\n".encode("utf-8"))
                last_ping = time.time()
                print("Sent PING")

            resp: str = sock.recv(2048).decode("utf-8").strip()

            if resp.startswith("PING"):
                sock.send("PONG\n".encode("utf-8"))
                last_ping = time.time()
                print("Sent PONG")

            elif len(resp) > 0:
                # print(f"Received message: {resp}")
                message: Optional[str] = parse_message(resp)
                if message:
                    message_queue.put(message)

    except KeyboardInterrupt:
        sock.close()
        sys.exit()
    except Exception as e:
        print(f"An error occurred: {e}")
        sock.close()
        sys.exit()


def main() -> None:
    print("Starting Twitch IRC client")

    # Initializes Twitch IRC thread
    irc_thread: Thread = Thread(target=init)
    irc_thread.daemon = True  # This allows the thread to exit when the main program does
    irc_thread.start()

    print("Starting main program loop")

    while True:
        try:
            message: str = message_queue.get(block=False)
            print(message)
        except queue.Empty:
            time.sleep(1)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
