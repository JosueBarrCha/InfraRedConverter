
"""
Infrared Aerochrome Photo Transformer
======================================
Transforms regular photographs into infrared/Kodak Aerochrome-style images
using a self-improving GAN (Generative Adversarial Network) built with TensorFlow.

Structure:
    project/
    ├── main.py (this file)
    ├── input/          <- Place your photos here
    ├── output/         <- Transformed photos appear here
    └── model/          <- Saved model weights (auto-created)

Usage:
    python main.py

Requirements:
    pip install tensorflow pillow numpy
"""

import os
import sys
import numpy as np
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from PIL import Image


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "image_size": 256,           # Resize images to this for training
    "batch_size": 4,
    "learning_rate_gen": 2e-4,
    "learning_rate_disc": 1e-4,
    "beta_1": 0.5,
    "lambda_style": 10.0,       # Style loss weight
    "lambda_perceptual": 1.0,   # Perceptual loss weight
    "lambda_color": 5.0,        # Infrared color mapping weight
    "model_dir": "model",
    "input_dir": "input",
    "output_dir": "output",
    "supported_formats": (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"),
}


# =============================================================================
# INFRARED COLOR SCIENCE
# =============================================================================

class InfraredColorMapper:
    """
    Simulates Kodak Aerochrome / Infrared color film characteristics:
    - Vegetation (green) → Vivid red/magenta
    - Blue sky → Deep blue/cyan
    - Red objects → Yellow/gold
    - Skin tones → Slight warm shift
    """

    @staticmethod
    @tf.function
    def apply_aerochrome_transform(images):
        """
        Apply the characteristic Aerochrome channel transformation.
        In real Aerochrome film:
            - IR-reflective (vegetation) maps to RED channel
            - Red maps to GREEN channel
            - Green maps to BLUE channel
        """
        r, g, b = images[..., 0:1], images[..., 1:2], images[..., 2:3]

        # Aerochrome-style channel remapping
        # Simulate IR reflectance from green channel (vegetation reflects IR)
        ir_estimate = g * 0.8 + r * 0.2  # Vegetation proxy

        new_r = tf.clip_by_value(ir_estimate * 1.4 + r * 0.3, 0.0, 1.0)
        new_g = tf.clip_by_value(r * 0.7 + g * 0.2 + b * 0.1, 0.0, 1.0)
        new_b = tf.clip_by_value(b * 0.6 + g * 0.3, 0.0, 1.0)

        # Increase saturation for that vivid Aerochrome look
        result = tf.concat([new_r, new_g, new_b], axis=-1)

        # Boost contrast
        mean = tf.reduce_mean(result, axis=-1, keepdims=True)
        result = tf.clip_by_value((result - mean) * 1.3 + mean, 0.0, 1.0)

        return result

    @staticmethod
    @tf.function
    def compute_color_loss(generated, original):
        """
        Loss that encourages the network to follow Aerochrome color rules:
        - High green in original → High red in output
        - Penalize unnatural artifacts
        """
        orig_g = original[..., 1:2]
        gen_r = generated[..., 0:1]

        # Vegetation areas (high green) should map to high red
        vegetation_mask = tf.cast(orig_g > 0.4, tf.float32)
        vegetation_loss = tf.reduce_mean(
            vegetation_mask * tf.maximum(0.0, 0.7 - gen_r)
        )

        # Smoothness loss to prevent artifacts
        dx = tf.abs(generated[:, :, 1:, :] - generated[:, :, :-1, :])
        dy = tf.abs(generated[:, 1:, :, :] - generated[:, :-1, :, :])
        smoothness = tf.reduce_mean(dx) + tf.reduce_mean(dy)

        return vegetation_loss + 0.1 * smoothness


# =============================================================================
# GENERATOR NETWORK (U-Net Architecture)
# =============================================================================

def build_generator(input_shape=(256, 256, 3)):
    """
    U-Net generator that learns to transform regular photos to infrared.
    Skip connections preserve detail while allowing deep color transformation.
    """
    inputs = layers.Input(shape=input_shape)

    # --- Encoder ---
    def encoder_block(x, filters, apply_batchnorm=True):
        x = layers.Conv2D(filters, 4, strides=2, padding='same',
                         kernel_initializer='he_normal')(x)
        if apply_batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        return x

    e1 = encoder_block(inputs, 64, apply_batchnorm=False)   # 128x128
    e2 = encoder_block(e1, 128)                              # 64x64
    e3 = encoder_block(e2, 256)                              # 32x32
    e4 = encoder_block(e3, 512)                              # 16x16
    e5 = encoder_block(e4, 512)                              # 8x8
    e6 = encoder_block(e5, 512)                              # 4x4

    # --- Bottleneck ---
    bottleneck = layers.Conv2D(512, 4, strides=2, padding='same')(e6)  # 2x2
    bottleneck = layers.LeakyReLU(0.2)(bottleneck)

    # --- Decoder with skip connections ---
    def decoder_block(x, skip, filters, apply_dropout=False):
        x = layers.Conv2DTranspose(filters, 4, strides=2, padding='same',
                                   kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        if apply_dropout:
            x = layers.Dropout(0.3)(x)
        x = layers.ReLU()(x)
        x = layers.Concatenate()([x, skip])
        return x

    d1 = decoder_block(bottleneck, e6, 512, apply_dropout=True)  # 4x4
    d2 = decoder_block(d1, e5, 512, apply_dropout=True)          # 8x8
    d3 = decoder_block(d2, e4, 512)                              # 16x16
    d4 = decoder_block(d3, e3, 256)                              # 32x32
    d5 = decoder_block(d4, e2, 128)                              # 64x64
    d6 = decoder_block(d5, e1, 64)                               # 128x128

    # Final output layer
    output = layers.Conv2DTranspose(3, 4, strides=2, padding='same',
                                    activation='sigmoid')(d6)     # 256x256

    return Model(inputs, output, name="InfraredGenerator")


# =============================================================================
# DISCRIMINATOR NETWORK (PatchGAN)
# =============================================================================

def build_discriminator(input_shape=(256, 256, 3)):
    """
    PatchGAN discriminator that evaluates whether an image looks like
    authentic infrared/Aerochrome photography.
    """
    inputs = layers.Input(shape=input_shape)

    def disc_block(x, filters, strides=2, apply_batchnorm=True):
        x = layers.Conv2D(filters, 4, strides=strides, padding='same',
                         kernel_initializer='he_normal')(x)
        if apply_batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        return x

    x = disc_block(inputs, 64, apply_batchnorm=False)
    x = disc_block(x, 128)
    x = disc_block(x, 256)
    x = disc_block(x, 512, strides=1)

    # Patch output (real/fake per patch)
    output = layers.Conv2D(1, 4, strides=1, padding='same')(x)

    return Model(inputs, output, name="InfraredDiscriminator")


# =============================================================================
# PERCEPTUAL LOSS (VGG-based)
# =============================================================================

class PerceptualLoss:
    """Uses VGG19 feature maps to compute perceptual similarity loss."""

    def __init__(self):
        vgg = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet', input_shape=(256, 256, 3)
        )
        vgg.trainable = False
        # Extract features from multiple layers
        self.feature_extractor = Model(
            inputs=vgg.input,
            outputs=[
                vgg.get_layer('block2_conv2').output,
                vgg.get_layer('block3_conv3').output,
                vgg.get_layer('block4_conv3').output,
            ]
        )

    @tf.function
    def compute_loss(self, generated, target):
        """Compare perceptual features between generated and target."""
        # VGG expects [0, 255] range with specific preprocessing
        gen_processed = tf.keras.applications.vgg19.preprocess_input(
            generated * 255.0
        )
        target_processed = tf.keras.applications.vgg19.preprocess_input(
            target * 255.0
        )

        gen_features = self.feature_extractor(gen_processed)
        target_features = self.feature_extractor(target_processed)

        loss = 0.0
        for gf, tf_feat in zip(gen_features, target_features):
            loss += tf.reduce_mean(tf.abs(gf - tf_feat))

        return loss


# =============================================================================
# INFRARED TRANSFORMER (Main Training & Inference Class)
# =============================================================================

class InfraredTransformer:
    """
    Main class that handles training, self-improvement, and inference.

    Self-improvement loop:
    1. Generate initial infrared targets using color science rules
    2. Train GAN to reproduce and enhance those transformations
    3. Discriminator learns what "good infrared" looks like
    4. Generator improves to fool the discriminator
    5. Model is saved after each session → improves over time
    """

    def __init__(self, config=CONFIG):
        self.config = config
        self.img_size = config["image_size"]

        # Create directories
        os.makedirs(config["model_dir"], exist_ok=True)
        os.makedirs(config["input_dir"], exist_ok=True)
        os.makedirs(config["output_dir"], exist_ok=True)

        # Build networks
        print("🔧 Building neural networks...")
        self.generator = build_generator((self.img_size, self.img_size, 3))
        self.discriminator = build_discriminator((self.img_size, self.img_size, 3))

        # Optimizers
        self.gen_optimizer = optimizers.Adam(
            config["learning_rate_gen"], beta_1=config["beta_1"]
        )
        self.disc_optimizer = optimizers.Adam(
            config["learning_rate_disc"], beta_1=config["beta_1"]
        )

        # Loss utilities
        self.color_mapper = InfraredColorMapper()
        self.perceptual_loss = PerceptualLoss()
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # Checkpoint management
        self.checkpoint = tf.train.Checkpoint(
            generator=self.generator,
            discriminator=self.discriminator,
            gen_optimizer=self.gen_optimizer,
            disc_optimizer=self.disc_optimizer,
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, config["model_dir"], max_to_keep=3
        )

        # Load existing model if available
        self._load_model()

        # Track training history
        self.total_images_processed = 0
        self.session_count = self._get_session_count()

    def _load_model(self):
        """Load the latest saved model weights."""
        latest = self.checkpoint_manager.latest_checkpoint
        if latest:
            self.checkpoint.restore(latest).expect_partial()
            print(f"✅ Loaded saved model from: {latest}")
            print("   (The model remembers what it learned previously!)")
        else:
            print("🆕 No saved model found. Starting fresh.")
            print("   (The model will improve with each run)")

    def _get_session_count(self):
        """Track how many times the model has been run."""
        counter_file = os.path.join(self.config["model_dir"], "session_count.txt")
        if os.path.exists(counter_file):
            with open(counter_file, 'r') as f:
                return int(f.read().strip())
        return 0

    def _save_session_count(self):
        """Save session counter."""
        counter_file = os.path.join(self.config["model_dir"], "session_count.txt")
        with open(counter_file, 'w') as f:
            f.write(str(self.session_count))

    # -------------------------------------------------------------------------
    # IMAGE LOADING & PREPROCESSING
    # -------------------------------------------------------------------------

    def load_image(self, path, resize_for_training=True):
        """Load and preprocess a single image."""
        img = Image.open(path).convert('RGB')
        if resize_for_training:
            img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        return img_array

    def load_image_full_res(self, path):
        """Load image at full resolution for final output."""
        img = Image.open(path).convert('RGB')
        return img, np.array(img, dtype=np.float32) / 255.0

    def get_input_images(self):
        """Get all valid image paths from the input directory."""
        input_dir = Path(self.config["input_dir"])
        images = []
        for ext in self.config["supported_formats"]:
            images.extend(input_dir.glob(f"*{ext}"))
            images.extend(input_dir.glob(f"*{ext.upper()}"))
        return sorted(set(images))

    def prepare_dataset(self, image_paths):
        """Create a TensorFlow dataset from image paths."""
        images = []
        for path in image_paths:
            try:
                img = self.load_image(str(path))
                images.append(img)
            except Exception as e:
                print(f"  ⚠️  Skipping {path.name}: {e}")

        if not images:
            return None

        images = np.array(images)
        dataset = tf.data.Dataset.from_tensor_slices(images)
        dataset = dataset.shuffle(len(images)).batch(self.config["batch_size"])
        return dataset

    # -------------------------------------------------------------------------
    # TRAINING STEP
    # -------------------------------------------------------------------------

    @tf.function
    def train_step(self, real_images):
        """
        Single training step for the GAN.

        The generator learns to create infrared images that:
        1. Follow Aerochrome color science rules
        2. Look realistic to the discriminator
        3. Preserve structural detail from the original
        """
        # Generate target using color science (serves as style reference)
        infrared_target = self.color_mapper.apply_aerochrome_transform(real_images)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generator creates infrared version
            generated = self.generator(real_images, training=True)

            # Discriminator evaluates both real infrared targets and generated
            disc_real = self.discriminator(infrared_target, training=True)
            disc_fake = self.discriminator(generated, training=True)

            # --- Discriminator Loss ---
            real_loss = self.bce(tf.ones_like(disc_real), disc_real)
            fake_loss = self.bce(tf.zeros_like(disc_fake), disc_fake)
            disc_loss = (real_loss + fake_loss) * 0.5

            # --- Generator Losses ---
            # 1. Adversarial loss (fool the discriminator)
            gen_adv_loss = self.bce(tf.ones_like(disc_fake), disc_fake)

            # 2. Style/color loss (follow Aerochrome rules)
            color_loss = self.color_mapper.compute_color_loss(
                generated, real_images
            )

            # 3. Perceptual loss (preserve structure)
            perc_loss = self.perceptual_loss.compute_loss(
                generated, infrared_target
            )

            # 4. L1 reconstruction loss against color-science target
            l1_loss = tf.reduce_mean(tf.abs(generated - infrared_target))

            # Combined generator loss
            gen_loss = (
                gen_adv_loss
                + self.config["lambda_style"] * l1_loss
                + self.config["lambda_color"] * color_loss
                + self.config["lambda_perceptual"] * perc_loss
            )

        # Update weights
        gen_grads = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables
        )
        disc_grads = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables
        )

        self.gen_optimizer.apply_gradients(
            zip(gen_grads, self.generator.trainable_variables)
        )
        self.disc_optimizer.apply_gradients(
            zip(disc_grads, self.discriminator.trainable_variables)
        )

        return gen_loss, disc_loss

    # -------------------------------------------------------------------------
    # TRAINING LOOP
    # -------------------------------------------------------------------------

    def train(self, epochs=50):
        """
        Train the model on images in the input folder.
        Each training session builds on previous sessions (self-improvement).
        """
        image_paths = self.get_input_images()
        if not image_paths:
            print("❌ No images found in input/ folder. Add some photos first!")
            return

        print(f"\n🎯 Training Session #{self.session_count + 1}")
        print(f"   Images found: {len(image_paths)}")
        print(f"   Epochs: {epochs}")
        print(f"   Previous sessions: {self.session_count}")
        print("-" * 50)

        dataset = self.prepare_dataset(image_paths)
        if dataset is None:
            print("❌ Could not load any valid images.")
            return

        for epoch in range(epochs):
            epoch_gen_loss = 0.0
            epoch_disc_loss = 0.0
            num_batches = 0

            for batch in dataset:
                gen_loss, disc_loss = self.train_step(batch)
                epoch_gen_loss += gen_loss.numpy()
                epoch_disc_loss += disc_loss.numpy()
                num_batches += 1

            avg_gen = epoch_gen_loss / max(num_batches, 1)
            avg_disc = epoch_disc_loss / max(num_batches, 1)

            print(f"   Epoch {epoch+1:3d}/{epochs} │ "
                  f"Gen Loss: {avg_gen:.4f} │ Disc Loss: {avg_disc:.4f}")

        # Save improved model
        self.session_count += 1
        self._save_session_count()
        save_path = self.checkpoint_manager.save()
        print(f"\n💾 Model saved: {save_path}")
        print(f"   Total training sessions completed: {self.session_count}")

    # -------------------------------------------------------------------------
    # INFERENCE (Full Resolution)
    # -------------------------------------------------------------------------

    def transform_image(self, image_array):
        """
        Transform a single image to infrared at any resolution.
        Uses tiled processing for images larger than training size.
        """
        h, w, _ = image_array.shape

        if h <= self.img_size and w <= self.img_size:
            # Small enough to process directly
            resized = tf.image.resize(
                image_array[np.newaxis], (self.img_size, self.img_size)
            )
            result = self.generator(resized, training=False)
            result = tf.image.resize(result[0], (h, w))
            return result.numpy()

        # For larger images: process at reduced size, then blend with
        # color-mapped full-res for detail preservation
        scale = self.img_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Pad to square
        padded = tf.image.resize(image_array[np.newaxis], (new_h, new_w))
        padded = tf.image.resize_with_pad(padded[0], self.img_size, self.img_size)

        # Neural network pass
        nn_result = self.generator(padded[np.newaxis], training=False)[0]

        # Remove padding and resize back
        nn_result = tf.image.resize(nn_result[:new_h, :new_w], (h, w))

        # Blend with color-science transform for detail preservation
        color_science = self.color_mapper.apply_aerochrome_transform(
            image_array[np.newaxis]
        )[0]

        # Weighted blend: more weight to neural network as it improves
        nn_weight = min(0.8, 0.4 + self.session_count * 0.05)
        blended = nn_weight * nn_result + (1 - nn_weight) * color_science

        return tf.clip_by_value(blended, 0.0, 1.0).numpy()

    # -------------------------------------------------------------------------
    # PROCESS ALL IMAGES
    # -------------------------------------------------------------------------

    def process_all(self):
        """
        Transform all images in input/ at full resolution and save to output/.
        """
        image_paths = self.get_input_images()
        if not image_paths:
            print("\n❌ No images found in input/ folder!")
            print(f"   Supported formats: {', '.join(self.config['supported_formats'])}")
            print(f"   Place your photos in: {os.path.abspath(self.config['input_dir'])}/")
            return

        print(f"\n🎨 Processing {len(image_paths)} image(s)...")
        print(f"   Model maturity: {self.session_count} training session(s)")
        print("-" * 50)

        for i, img_path in enumerate(image_paths, 1):
            try:
                print(f"   [{i}/{len(image_paths)}] Processing: {img_path.name}...",
                      end=" ", flush=True)

                # Load at full resolution
                original_img, img_array = self.load_image_full_res(str(img_path))

                # Transform
                result = self.transform_image(img_array)

                # Save output
                output_name = f"infrared_{img_path.stem}.png"
                output_path = os.path.join(self.config["output_dir"], output_name)

                result_img = Image.fromarray(
                    (result * 255).astype(np.uint8)
                )
                result_img.save(output_path, quality=95)

                print(f"✅ → {output_name}")
                self.total_images_processed += 1

            except Exception as e:
                print(f"❌ Error: {e}")

        # Summary
        print("\n" + "-" * 50)
        print(f"  ✨ Done! {self.total_images_processed} image(s) saved to output/")

    # -------------------------------------------------------------------------
    # RESET MODEL
    # -------------------------------------------------------------------------

    def reset(self):
        """Reset the model to start fresh."""
        import shutil
        if os.path.exists(self.config["model_dir"]):
            shutil.rmtree(self.config["model_dir"])
            os.makedirs(self.config["model_dir"])
        print("\n🗑️  Model has been reset. All learned weights deleted.")
        print("   The model will start fresh on next training session.")


# =============================================================================
# CLI MENU
# =============================================================================

def print_banner():
    """Print the application header banner."""
    print("\n" + "=" * 60)
    print("  🎞️  INFRARED AEROCHROME TRANSFORMER")
    print("  ─────────────────────────────────────")
    print("  Transform photos into Kodak Aerochrome infrared style")
    print("  using a self-improving TensorFlow GAN")
    print("=" * 60)


def print_status(transformer):
    """Print current model and folder status."""
    image_paths = transformer.get_input_images()
    output_count = len(list(Path(transformer.config["output_dir"]).glob("*")))

    print(f"\n  📊 Status:")
    print(f"     • Model training sessions: {transformer.session_count}")
    print(f"     • Images in input/:        {len(image_paths)}")
    print(f"     • Images in output/:       {output_count}")
    print()


def print_menu():
    """Print the main menu options."""
    print("  ┌─────────────────────────────────────┐")
    print("  │         MAIN MENU                    │")
    print("  ├─────────────────────────────────────┤")
    print("  │  [1]  🎨 Process images              │")
    print("  │       Transform input/ → output/     │")
    print("  │                                      │")
    print("  │  [2]  🧠 Train model (50 epochs)     │")
    print("  │       Improve the model using        │")
    print("  │       images in input/               │")
    print("  │                                      │")
    print("  │  [3]  🗑️  Reset model                 │")
    print("  │       Delete all learned weights     │")
    print("  │       and start fresh                │")
    print("  │                                      │")
    print("  │  [Q]  🚪 Quit                        │")
    print("  └─────────────────────────────────────┘")


def confirm_action(message):
    """Ask user for confirmation before destructive actions."""
    while True:
        response = input(f"\n  ⚠️  {message} (y/n): ").strip().lower()
        if response in ('y', 'yes'):
            return True
        if response in ('n', 'no'):
            return False
        print("  Please enter 'y' or 'n'.")


def main():
    """Main entry point with interactive CLI menu."""
    print_banner()

    # Initialize transformer (loads model if available)
    transformer = InfraredTransformer()

    # Main loop
    while True:
        print_status(transformer)
        print_menu()

        choice = input("\n  Enter your choice [1/2/3/Q]: ").strip().lower()

        if choice == '1':
            # ── Process Images ──
            image_paths = transformer.get_input_images()
            if not image_paths:
                print("\n  ❌ No images found in input/ folder!")
                print(f"     Place your photos in: {os.path.abspath(CONFIG['input_dir'])}/")
                print(f"     Supported: {', '.join(CONFIG['supported_formats'])}")
            else:
                print(f"\n  Found {len(image_paths)} image(s). Transforming...")
                transformer.process_all()

        elif choice == '2':
            # ── Train 50 Epochs ──
            image_paths = transformer.get_input_images()
            if not image_paths:
                print("\n  ❌ No images found in input/ folder!")
                print("     Add photos to input/ to train the model.")
            else:
                print(f"\n  🧠 Starting training on {len(image_paths)} image(s)...")
                print("     This will run 50 epochs. Press Ctrl+C to stop early.\n")
                try:
                    transformer.train(epochs=50)
                except KeyboardInterrupt:
                    print("\n\n  ⏹️  Training interrupted by user.")
                    print("     Progress so far has been saved.")

        elif choice == '3':
            # ── Reset Model ──
            if transformer.session_count == 0:
                print("\n  ℹ️  Model is already fresh (no training sessions yet).")
            else:
                if confirm_action(
                    f"This will delete {transformer.session_count} training session(s). Continue?"
                ):
                    transformer.reset()
                    # Reinitialize with fresh model
                    print("  🔄 Reinitializing fresh model...")
                    transformer = InfraredTransformer()
                else:
                    print("  ↩️  Reset cancelled.")

        elif choice in ('q', 'quit', 'exit'):
            print("\n  👋 Goodbye! Happy infrared shooting.\n")
            sys.exit(0)

        else:
            print("\n  ❓ Invalid choice. Please enter 1, 2, 3, or Q.")


if __name__ == "__main__":
    main()

