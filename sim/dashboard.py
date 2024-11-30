import gradio as gr
import hydra
from omegaconf import DictConfig

from definitions import ROOT_PATH


def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)

@hydra.main(config_name="dashboard", config_path=str(ROOT_PATH / "config"))
def main(cfg: DictConfig):
    demo = gr.Interface(
        fn=greet,
        inputs=["text", "slider"],
        outputs=["text"],
    )

    demo.launch()


if __name__ == "__main__":
    main()


