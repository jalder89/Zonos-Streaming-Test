import time

import torch
import torchaudio

from zonos.model import Zonos

texts = [
    "The old clock tower hadn't chimed in living memory.",
    "Its stone face, weathered and stained, watched over the perpetually drowsy town.",
    "Elara, however, felt a strange pull towards it.",
    "She often sketched its silhouette in her worn notebook.",
    "One moonless night, a faint, melodic hum vibrated through the cobblestones beneath her feet.",
    "It seemed to emanate from the silent tower.",
    "Driven by a curiosity stronger than fear, she crept towards the heavy oak door.",
    "Surprisingly, it swung open at her touch, revealing a spiral staircase choked with dust.",
    "The air inside was thick with the scent of ozone and something ancient.",
    "She ascended, each step echoing in the profound stillness.",
    "Higher and higher she climbed, the humming growing louder, resonating within her chest.",
    "Finally, she reached the belfry.",
    "Instead of bells, intricate crystalline structures pulsed with soft, blue light.",
    "They hung suspended, rotating slowly, emitting the enchanting melody.",
    "In the center hovered a sphere of swirling energy.",
    "As Elara approached, the humming intensified, the light brightening.",
    "Tendrils of energy reached out from the sphere, brushing against her fingertips.",
    "A flood of images poured into her mind: star charts, forgotten equations, galaxies blooming and dying.",
    "She wasn't just in a clock tower; she was inside a celestial resonator.",
    "It was a device left by travelers from a distant star, waiting for someone attuned to its frequency.",
    "Elara realized the tower hadn't been silent, just waiting.",
    "She raised her hands, not in fear, but in acceptance.",
    "The energy flowed into her, cool and invigorating.",
    "Suddenly, with a resonant *gong*, the tower chimed, a sound unheard for centuries.",
    "Its song wasn't marking time, but awakening possibilities across the cosmos.",
]


def main():
    """
    FIXED VERSION: Enhanced streaming example with proper error handling and debugging.
    """
    # Use CUDA if available.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        # Load the model (here we use the transformer variant).
        print("Loading model...")
        model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
        model.requires_grad_(False).eval()
        print(f"Model loaded successfully on {device}")

        # Load a reference speaker audio to generate a speaker embedding.
        print("Loading reference audio...")
        wav, sr = torchaudio.load("assets/exampleaudio.mp3")
        speaker = model.make_speaker_embedding(wav, sr)
        print(f"Speaker embedding created: shape={speaker.shape}")

        # Set a random seed for reproducibility.
        torch.manual_seed(777)

        # Accumulate audio chunks as they are generated.
        audio_chunks = []
        t0 = time.time()
        generated = 0
        ttfb = None
        chunk_count = 0

        def generator():
            # Can stream from your LLM or other source here, just partition the text into
            # sentences with nltk or rule based tokenizer. See example here:
            # https://stackoverflow.com/a/31505798
            for i, text in enumerate(texts):
                elapsed = int((time.time() - t0) * 1000)
                print(f"Yielding sentence {i+1}/{len(texts)} at {elapsed}ms: {text[:50]}...")
                yield {
                    "text": text,
                    "speaker": speaker,
                    "language": "en-us",
                }

        # --- STREAMING GENERATION ---
        print("Starting streaming generation...")

        # FIXED: More conservative chunk schedule to reduce memory pressure
        # Start with smaller chunks and gradually increase
        stream_generator = model.stream(
            cond_dicts_generator=generator(),
            chunk_schedule=[8, 12, 16, 20, 25, 30],  # More conservative schedule
            chunk_overlap=1,  # Reduced overlap to minimize complexity
            max_new_tokens=512,  # Reduced from default for testing
            mark_boundaries=True,
        )

        print("Stream generator created, starting iteration...")

        for i, audio_chunk in enumerate(stream_generator):
            try:
                if isinstance(audio_chunk, str):
                    print(f"Sentence boundary: {audio_chunk[:50]}...")
                    continue

                # Validate audio chunk
                if audio_chunk is None or audio_chunk.numel() == 0:
                    print(f"Chunk {i + 1}: Empty audio chunk received")
                    continue

                chunk_count += 1
                audio_chunks.append(audio_chunk.cpu())  # Move to CPU for storage
                
                elapsed = int((time.time() - t0) * 1000)
                if ttfb is None:
                    ttfb = elapsed
                    print(f"Time to first byte: {ttfb}ms")
                
                chunk_duration = audio_chunk.shape[-1] / 44100 * 1000  # Duration in ms
                generated += chunk_duration
                
                gap = "GAP" if ttfb + generated < elapsed else ""
                print(f"Chunk {chunk_count:>3}: elapsed {elapsed:>5}ms | "
                      f"chunk_duration {chunk_duration:>5.0f}ms | "
                      f"total_generated {ttfb + generated:>5.0f}ms | "
                      f"chunk_shape {audio_chunk.shape} {gap}")

                # Clear GPU memory periodically
                if chunk_count % 5 == 0 and device == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing chunk {i}: {e}")
                continue

    except Exception as e:
        print(f"Fatal error during streaming: {e}")
        import traceback
        traceback.print_exc()
        return

    try:
        if audio_chunks:
            # Concatenate all audio chunks along the time axis.
            print(f"Concatenating {len(audio_chunks)} audio chunks...")
            audio = torch.cat(audio_chunks, dim=-1)

            generation_time = round(time.time() - t0, 3)
            audio_duration = round(audio.shape[-1] / 44100, 3)
            rtx = round(audio_duration / generation_time, 2) if generation_time > 0 else 0

            print(f"Generation complete:")
            print(f"  TTFB: {ttfb}ms")
            print(f"  Generation time: {generation_time}s")
            print(f"  Audio duration: {audio_duration}s")
            print(f"  Real-time factor: {rtx}x")
            print(f"  Final audio shape: {audio.shape}")

            # Save the full audio as a WAV file.
            out_sr = model.autoencoder.sampling_rate
            output_file = "streaming_fixed.wav"
            torchaudio.save(output_file, audio, out_sr)
            print(f"Saved streaming audio to '{output_file}' (sampling rate: {out_sr} Hz)")
        else:
            print("No audio chunks were generated successfully")

    except Exception as e:
        print(f"Error during final processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()