import threading
import queue
import asyncio
from time import sleep

from flow_prompt.services.flow_prompt import FlowPromptService


class SaveWorker:
    def __init__(self):
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self.worker)
        self.thread.daemon = True  # Daemon thread exits when main program exits
        self.thread.start()

    def save_user_interaction_async(
        self, api_token, prompt_data, context, response, metrics
    ):
        FlowPromptService.save_user_interaction(
            api_token, prompt_data, context, response, metrics
        )

    def worker(self):
        while True:
            task = self.queue.get()
            if task is None:
                sleep(1)
                continue
            api_token, prompt_data, context, response, metrics = task
            self.save_user_interaction_async(
                api_token, prompt_data, context, response, metrics
            )
            self.queue.task_done()

    def add_task(self, api_token, prompt_data, context, response, metrics={}):
        self.queue.put((api_token, prompt_data, context, response, metrics))
