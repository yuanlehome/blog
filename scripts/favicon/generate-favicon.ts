import fs from 'fs';
import path from 'path';
import sharp from 'sharp';
import pngToIco from 'png-to-ico';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const SOURCE_IMAGE = path.join(__dirname, '../../public/placeholder-social.jpg');
const OUTPUT_DIR = path.join(__dirname, '../../public');

interface FaviconConfig {
  name: string;
  size: number;
}

const configs: FaviconConfig[] = [
  { name: 'favicon-32x32.png', size: 32 },
  { name: 'favicon-16x16.png', size: 16 },
  { name: 'apple-touch-icon.png', size: 180 },
];

async function generateFavicons() {
  console.log('🎨 Generating favicons from public/placeholder-social.jpg...');

  // Check if source image exists
  if (!fs.existsSync(SOURCE_IMAGE)) {
    throw new Error(`Source image not found: ${SOURCE_IMAGE}`);
  }

  // Generate PNG favicons
  for (const config of configs) {
    const outputPath = path.join(OUTPUT_DIR, config.name);
    console.log(`  → Generating ${config.name} (${config.size}x${config.size})`);

    await sharp(SOURCE_IMAGE)
      .resize(config.size, config.size, {
        fit: 'cover',
        position: 'center',
      })
      .png()
      .toFile(outputPath);

    console.log(`  ✓ Created ${config.name}`);
  }

  // Generate favicon.ico from the 32x32 PNG
  const favicon32Path = path.join(OUTPUT_DIR, 'favicon-32x32.png');
  const favicon16Path = path.join(OUTPUT_DIR, 'favicon-16x16.png');
  const icoPath = path.join(OUTPUT_DIR, 'favicon.ico');

  // Verify PNG files exist before creating ICO
  if (!fs.existsSync(favicon32Path) || !fs.existsSync(favicon16Path)) {
    throw new Error('Required PNG files for ICO generation are missing');
  }

  console.log('  → Generating favicon.ico');
  const icoBuffer = await pngToIco([favicon32Path, favicon16Path]);
  fs.writeFileSync(icoPath, icoBuffer);
  console.log('  ✓ Created favicon.ico');

  console.log('✨ All favicons generated successfully!');
}

// Run the script
generateFavicons().catch((error) => {
  console.error('❌ Error generating favicons:', error);
  process.exit(1);
});
