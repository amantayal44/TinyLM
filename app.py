import gradio as gr
import argparse
import subprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to the model file')
    parser.add_argument('--tokenizer', type=str, required=True, help='Path to the tokenizer model file')
    parser.add_argument('--gradio', action='store_true', default=False, help='Run gradio interface')
    parser.add_argument('--share', action='store_true', default=False, help='Create a public link for sharing')
    args = parser.parse_args()

    # Create text generation interface
    def generate_text(text: str, temp=1.0, top_k=5, max_len=100) -> str:
        cmd = [
            'python3', 'inference.py',
            '--model', args.model,
            '--tokenizer', args.tokenizer,
            '--text', text,
            '--temp', str(temp),
            '--top_k', str(top_k),
            '--max_len', str(max_len),
            '--num_samples', '1',
        ]
        print(f'Running command: {" ".join(cmd)}')
        result = subprocess.run(cmd, stdout=subprocess.PIPE)

        if result.returncode != 0:
            print(f'Error generating text: {result.stderr}')
            return 'Error generating text'
        output = result.stdout.decode('utf-8').strip()
        output = output.split(':', 1)[-1]
        # remove quotes from start and end
        output = output.strip()[1:-1]
        # convert escaped characters like \n, \t to actual characters
        output = output.encode().decode('unicode_escape')
        print(f'Generated text: {output}')
        return output
    
    if args.gradio:
        # Create gradio interface
        iface = gr.Interface(fn=generate_text,
                inputs=[
                    gr.Textbox(lines=5, placeholder="Enter text here...", label="Input Text"),
                    gr.Slider(minimum=0.1, maximum=10.0, step=0.1, value=1.0, label="Temperature"),
                    gr.Slider(minimum=1, maximum=100, step=1, value=5, label="Top K"),
                    gr.Slider(minimum=10, maximum=255, step=10, value=100, label="Max Length")
                ],
                outputs=gr.Textbox(label="Generated Text"),
                title="TinyLM Text Generation",
                description="Generate stories using TinyLM model",
                allow_flagging="never",
            )
        iface.launch(share=args.share)
    
    else:
        # Run inference in command line
        while True:
            text = input('Enter text: ')
            temp = float(input('Temperature: '))
            top_k = int(input('Top K: '))
            max_len = int(input('Max Length: '))
            generate_text(text, temp, top_k, max_len)