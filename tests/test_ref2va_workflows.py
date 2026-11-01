# SPDX-License-Identifier: GPL-3.0-only
"""CPU-only structural checks for the portable Ref2VA API workflows."""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "ref2va-official-768p-10s-api.json": (1344, 768, False),
    "ref2va-official-480p-10s-api.json": (864, 480, False),
    "ref2va-standalone-audio-480p-10s-api.json": (864, 480, True),
}
# Original showcase prompt before the Comfy API marker conversion.
OFFICIAL_PROMPT = ("References: @Image1 is the character sheet (front, three-quarter and back views of ONE character). @Image2 is the environment. @Video1 is the motion reference only. Make a new video of the character from @Image1, inside the environment from @Image2, performing exactly the body motion, choreography, timing and camera movement of the person in @Video1. Take NOTHING else from @Video1 - not the person's face, body, clothes, props, location, background, colours or photographic look. Keep the character exactly as designed in @Image1 - same face, proportions, outfit, colours - and render the whole video in the exact art style of @Image1 and @Image2, never photoreal. Only ONE character in frame. Style: vintage pop-art fashion illustration with black ink outlines, flat colour blocks, halftone and risograph grain, exactly as @Image1 and @Image2. Desmond dances on the terracotta checkerboard floor of his record shop from @Image2: the same whirling spins, flung arms and kicks as in @Video1, his mustard suit jacket flaring with each turn. Where the person in @Video1 holds the potted fern overhead, Desmond holds his vinyl record high above his head like a dance partner. He ends with arms wide and a little deadpan bow. Audio: an original upbeat 1960s-style soul groove playing from the turntable, loafers tapping and squeaking on the floor, a soft record-crackle. Normal real-time speed, NO slow motion. No text, no captions, no logos.")


class Ref2VAWorkflows(unittest.TestCase):
    def test_workflows(self):
        for name, (width, height, standalone) in CASES.items():
            with self.subTest(workflow=name):
                graph = json.loads((ROOT / 'workflows' / name).read_text(encoding='utf-8'))
                expected_classes = {
                    '119': 'VAELoader', '120': 'VAELoader', '127': 'UNETLoader',
                    '128': 'CLIPLoader', '134': 'PikaH3LowPeakLoRA', '201': 'PikaH3Optimize',
                    '149': 'LoadImage', '150': 'LoadImage', '151': 'LoadVideo',
                    '152': 'GetVideoComponents', '136': 'MiniMaxH3ReferenceToVideo',
                    '126': 'BasicGuider', '129': 'RandomNoise', '123': 'KSamplerSelect',
                    '143': 'ManualSigmas', '125': 'SamplerCustomAdvanced',
                    '121': 'VAEDecodeAudio', '122': 'VAEDecode', '130': 'CreateVideo',
                    '92': 'SaveVideo',
                }
                if standalone:
                    expected_classes['153'] = 'LoadAudio'
                self.assertEqual({k: v['class_type'] for k, v in graph.items()}, expected_classes)
                inputs = lambda node: graph[node]['inputs']
                self.assertEqual(inputs('119'), {'vae_name': 'minimax_h3_video_vae_int8_convrot.safetensors'})
                self.assertEqual(inputs('120'), {'vae_name': 'minimax_h3_audio_vae_fp32.safetensors'})
                self.assertEqual(inputs('127'), {'unet_name': 'minimax_h3_ref2va_pruned_int8_convrot.safetensors', 'weight_dtype': 'default'})
                self.assertEqual(inputs('128'), {'clip_name': 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors', 'type': 'minimax', 'device': 'default'})
                self.assertEqual(inputs('134'), {'model': ['127', 0], 'lora_name': 'pika_h3_flex_ref2va_v1_bf16.safetensors', 'strength': 1.0, 'mode': 'low_peak'})
                self.assertEqual(inputs('201'), {'model': ['134', 0], 'chunk_size': '16384', 'activation': 'native'})
                self.assertEqual(inputs('149'), {'image': 'pika_official_character_ref.png'})
                self.assertEqual(inputs('150'), {'image': 'pika_official_environment_ref.png'})
                self.assertEqual(inputs('151'), {'file': 'pika_official_motion_ref.mp4'})
                self.assertEqual(inputs('152'), {'video': ['151', 0]})
                prompt = OFFICIAL_PROMPT.replace('@Image1', '<Picture 1>').replace('@Image2', '<Picture 2>').replace('@Video1', '<Video 1>')
                expected_ref = {'clip': ['128', 0], 'vae': ['119', 0], 'audio_vae': ['120', 0],
                    'ref_images.ref_image_0': ['149', 0], 'ref_images.ref_image_1': ['150', 0],
                    'ref_videos.ref_video_0': ['152', 0], 'prompt': prompt,
                    'width': width, 'height': height, 'length': 243, 'ref_image_size': 'match'}
                if standalone:
                    expected_ref['ref_audios.ref_audio_0'] = ['153', 0]
                    self.assertEqual(inputs('153'), {'audio': 'pika_official_motion_audio_standalone.wav'})
                else:
                    expected_ref['ref_video_audios.ref_video_audio_0'] = ['152', 1]
                self.assertEqual(inputs('136'), expected_ref)
                self.assertEqual(inputs('126'), {'model': ['201', 0], 'conditioning': ['136', 0]})
                self.assertEqual(inputs('129'), {'noise_seed': 42})
                self.assertEqual(inputs('123'), {'sampler_name': 'lcm'})
                self.assertEqual(inputs('143'), {'sigmas': '1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0'})
                self.assertEqual(inputs('125'), {'noise': ['129', 0], 'guider': ['126', 0], 'sampler': ['123', 0], 'sigmas': ['143', 0], 'latent_image': ['136', 1]})
                self.assertEqual(inputs('121'), {'samples': ['125', 0], 'vae': ['120', 0]})
                self.assertEqual(inputs('122'), {'samples': ['125', 0], 'vae': ['119', 0]})
                self.assertEqual(inputs('130'), {'images': ['122', 0], 'audio': ['121', 0], 'fps': 24.0, 'bit_depth': 'auto', 'color_space': 'sRGB', 'codec': 'none'})
                prefix = name.removeprefix('ref2va-').removesuffix('-api.json').replace('-', '_')
                self.assertEqual(inputs('92'), {'video': ['130', 0], 'filename_prefix': 'pika_ref2va_' + prefix, 'format': 'mp4', 'codec': {'codec': 'h264'}})
                # Scan every string value recursively, not just known filename fields.
                def check_paths(value):
                    if isinstance(value, str):
                        self.assertNotRegex(value, r'(?i)(?:[a-z]:[\\/]|\\\\|(?:^|\s)/[^\s/]+(?:/|$))')
                    elif isinstance(value, dict):
                        for key, item in value.items():
                            check_paths(key)
                            check_paths(item)
                    elif isinstance(value, list):
                        for item in value:
                            check_paths(item)
                check_paths(graph)


if __name__ == '__main__':
    unittest.main()
