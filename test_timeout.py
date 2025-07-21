import time
from datetime import timedelta

import click
from accelerate import Accelerator, InitProcessGroupKwargs
from torch import tensor


@click.command()
@click.argument("wait_for", type=int, default=8)
def main(wait_for):
    kwargs = [InitProcessGroupKwargs(timeout=timedelta(seconds=10))]
    accelerator = Accelerator(kwargs_handlers=kwargs)

    if accelerator.is_main_process:
        t = tensor(0).to(accelerator.device)
        time.sleep(wait_for)
    else:
        t = tensor(0).to(accelerator.device)
    accelerator.wait_for_everyone()

    print("All called!")


if __name__ == "__main__":
    main()
