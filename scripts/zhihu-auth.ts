/**
 * Zhihu Authentication Script
 *
 * Generates a Playwright storageState JSON file for authenticated Zhihu access.
 * This allows CI to import Zhihu articles without encountering login walls.
 *
 * Usage:
 *   npm run zhihu:auth
 *
 * After login, the storage state is saved to zhihu-storage-state.json.
 * Upload this file's base64-encoded content to GitHub Secrets as ZHIHU_STORAGE_STATE_B64.
 */

import { chromium } from '@playwright/test';
import path from 'path';
import readline from 'readline';

const STORAGE_STATE_PATH = 'zhihu-storage-state.json';
const ZHIHU_SIGNIN_URL = 'https://www.zhihu.com/signin';

function promptForEnter(message: string): Promise<void> {
  return new Promise((resolve) => {
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });
    rl.question(message, () => {
      rl.close();
      resolve();
    });
  });
}

async function main() {
  console.log('='.repeat(60));
  console.log('Zhihu Authentication - Generate Storage State');
  console.log('='.repeat(60));
  console.log('');
  console.log('This script will open a browser window for you to log in to Zhihu.');
  console.log('After you complete the login process, press Enter in this terminal.');
  console.log('');

  const browser = await chromium.launch({
    headless: false,
  });

  const context = await browser.newContext({
    locale: 'zh-CN',
    viewport: { width: 1280, height: 720 },
  });

  const page = await context.newPage();

  try {
    console.log(`Opening ${ZHIHU_SIGNIN_URL}...`);
    await page.goto(ZHIHU_SIGNIN_URL, { waitUntil: 'domcontentloaded', timeout: 60000 });

    console.log('');
    console.log('-'.repeat(60));
    console.log('请在浏览器中完成登录，登录成功后按回车键继续...');
    console.log('(Please complete the login in the browser, then press Enter to continue...)');
    console.log('-'.repeat(60));

    await promptForEnter('');

    // Verify login by checking for Zhihu authentication cookies
    // z_c0: Main session token that indicates a logged-in user
    // KLBRSID: Session identifier used for request tracking
    // Note: Zhihu's cookie names may change; if this validation fails but login works, update these names
    const currentUrl = page.url();
    const cookies = await context.cookies();
    const hasZhihuCookies = cookies.some(
      (c) => c.domain.includes('zhihu.com') && (c.name === 'z_c0' || c.name === 'KLBRSID'),
    );

    if (!hasZhihuCookies) {
      console.warn('');
      console.warn('Warning: No Zhihu authentication cookies detected.');
      console.warn('The login may not have completed successfully.');
      console.warn('Continuing to save storage state anyway...');
    }

    // Save storage state
    const storagePath = path.resolve(STORAGE_STATE_PATH);
    await context.storageState({ path: storagePath });

    console.log('');
    console.log('='.repeat(60));
    console.log('SUCCESS! Storage state saved to:');
    console.log(`  ${storagePath}`);
    console.log('');
    console.log('Next steps:');
    console.log('1. Encode the file to base64:');
    console.log(`   base64 -w 0 ${STORAGE_STATE_PATH} > zhihu-storage-state.b64`);
    console.log('');
    console.log('2. Copy the base64 content and add it as a GitHub Secret:');
    console.log('   Secret name: ZHIHU_STORAGE_STATE_B64');
    console.log('');
    console.log('3. When the storage state expires (login wall appears),');
    console.log('   re-run this script and update the secret.');
    console.log('='.repeat(60));
    console.log(`Current URL: ${currentUrl}`);
    console.log(`Cookies found: ${cookies.length}`);
  } catch (error) {
    console.error('Error during authentication:', error);
    process.exit(1);
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
