#!/usr/bin/env tsx
/**
 * Config Effectiveness Audit Script
 *
 * This script analyzes the entire codebase to ensure all YAML config items
 * are actually used and not shadowed by hardcoded values.
 *
 * Usage: npm run config:audit
 */

import * as fs from 'fs';
import * as path from 'path';

/**
 * Recursively find files matching patterns
 */
function findFiles(dir: string, patterns: string[]): string[] {
  const results: string[] = [];
  const entries = fs.readdirSync(dir, { withFileTypes: true });

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);

    if (entry.isDirectory()) {
      if (entry.name !== 'node_modules' && entry.name !== '.git') {
        results.push(...findFiles(fullPath, patterns));
      }
    } else {
      if (
        patterns.some((p) => entry.name.endsWith(p)) &&
        !entry.name.includes('.test.') &&
        !entry.name.includes('.spec.')
      ) {
        results.push(fullPath);
      }
    }
  }

  return results;
}

interface ConfigItem {
  path: string; // e.g., "layout.alignment.footerAlign"
  type: string; // e.g., "enum", "string", "number"
  possibleValues?: string[]; // For enums
  defaultValue: string | number | boolean;
  file: string; // YAML file name
  status: 'USED' | 'READ_ONLY' | 'SHADOWED' | 'UNUSED';
  usages: string[]; // File paths where it's used
  issues: string[]; // Any problems found
}

interface AuditResult {
  inventory: ConfigItem[];
  summary: {
    total: number;
    used: number;
    readOnly: number;
    shadowed: number;
    unused: number;
  };
  exitCode: number; // 0 = OK, 1 = problems found
}

/**
 * Extract config items from schema files
 */
function extractConfigInventory(): ConfigItem[] {
  // For now, manually define known config items that need checking
  // Future: Parse schema files automatically to extract all config items
  const knownConfigs: ConfigItem[] = [
    {
      path: 'layout.alignment.headerAlign',
      type: 'enum',
      possibleValues: ['left', 'center'],
      defaultValue: 'left',
      file: 'layout.yml',
      status: 'UNUSED',
      usages: [],
      issues: [],
    },
    {
      path: 'layout.alignment.footerAlign',
      type: 'enum',
      possibleValues: ['left', 'center'],
      defaultValue: 'left',
      file: 'layout.yml',
      status: 'UNUSED',
      usages: [],
      issues: [],
    },
    {
      path: 'layout.alignment.postMetaAlign',
      type: 'enum',
      possibleValues: ['left', 'center'],
      defaultValue: 'left',
      file: 'layout.yml',
      status: 'UNUSED',
      usages: [],
      issues: [],
    },
    {
      path: 'layout.layoutMode',
      type: 'enum',
      possibleValues: ['centered', 'rightSidebar', 'leftSidebar'],
      defaultValue: 'rightSidebar',
      file: 'layout.yml',
      status: 'UNUSED',
      usages: [],
      issues: [],
    },
    {
      path: 'layout.sidebar.enabled',
      type: 'boolean',
      defaultValue: true,
      file: 'layout.yml',
      status: 'UNUSED',
      usages: [],
      issues: [],
    },
    {
      path: 'layout.sidebar.position',
      type: 'enum',
      possibleValues: ['left', 'right'],
      defaultValue: 'right',
      file: 'layout.yml',
      status: 'UNUSED',
      usages: [],
      issues: [],
    },
  ];

  return knownConfigs;
}

/**
 * Scan source files for config usage
 */
function scanConfigUsage(inventory: ConfigItem[]): void {
  const srcDir = path.join(process.cwd(), 'src');
  const sourceFiles = findFiles(srcDir, ['.astro', '.ts', '.tsx', '.css']);

  for (const item of inventory) {
    // Extract the key from the path (e.g., "footerAlign" from "layout.alignment.footerAlign")
    const keyParts = item.path.split('.');
    const key = keyParts[keyParts.length - 1];

    let hasProperUsage = false;
    let hasUtilityMapping = false;

    for (const file of sourceFiles) {
      const content = fs.readFileSync(file, 'utf-8');

      // Check if this config key is referenced
      if (content.includes(key)) {
        item.usages.push(file);

        // Check if it's properly used through utility functions
        if (
          key.includes('Align') &&
          (content.includes('alignToTextClass') ||
            content.includes('alignToJustifyClass') ||
            content.includes('alignToItemsClass') ||
            content.includes('getAllAlignmentClasses'))
        ) {
          hasUtilityMapping = true;
        }
      }

      // For alignment configs, check if components/pages consume them through utility functions
      if (key.includes('Align')) {
        // Check if alignment utility is imported
        if (content.includes('from ') && content.includes('/lib/ui/alignment')) {
          hasProperUsage = true;
        }

        // Still check for hardcoded values that might shadow config
        if (item.type === 'enum' && item.possibleValues) {
          for (const value of item.possibleValues) {
            const classPatterns = [
              `text-${value}`,
              `justify-${value === 'left' ? 'start' : value}`,
              `items-${value === 'left' ? 'start' : value}`,
            ];

            for (const pattern of classPatterns) {
              // Only flag as issue if hardcoded AND no utility mapping
              if (content.includes(pattern) && !hasUtilityMapping && !content.includes('alignTo')) {
                item.issues.push(
                  `Hardcoded class "${pattern}" found in ${file} - may shadow config`,
                );
              }
            }
          }
        }
      }
    }

    // Determine status
    if (item.usages.length === 0) {
      item.status = 'UNUSED';
    } else if (hasProperUsage || hasUtilityMapping) {
      item.status = 'USED';
    } else if (item.issues.length > 0) {
      item.status = 'SHADOWED';
    } else {
      // Need to verify if it actually affects rendering
      item.status = 'READ_ONLY';
    }
  }
}

/**
 * Generate audit report
 */
function generateReport(result: AuditResult): string {
  const lines: string[] = [];

  lines.push('# 配置生效性审计报告 (Config Effectiveness Audit Report)');
  lines.push('');
  lines.push(`生成时间: ${new Date().toISOString()}`);
  lines.push('');

  lines.push('## 一、审计摘要 (Summary)');
  lines.push('');
  lines.push(`- **总配置项**: ${result.summary.total}`);
  lines.push(`- **已生效 (USED)**: ${result.summary.used}`);
  lines.push(`- **仅读取未影响渲染 (READ_ONLY)**: ${result.summary.readOnly}`);
  lines.push(`- **被硬编码覆盖 (SHADOWED)**: ${result.summary.shadowed}`);
  lines.push(`- **完全未使用 (UNUSED)**: ${result.summary.unused}`);
  lines.push('');

  if (result.summary.shadowed > 0 || result.summary.unused > 0) {
    lines.push('⚠️ **发现问题**: 存在未生效或被覆盖的配置项，需要修复！');
  } else {
    lines.push('✅ **状态良好**: 所有配置项都在正常使用中。');
  }
  lines.push('');

  lines.push('## 二、配置项清单 (Config Inventory)');
  lines.push('');
  lines.push('| 配置路径 | 类型 | 默认值 | 状态 | 问题 |');
  lines.push('|---------|------|--------|------|------|');

  for (const item of result.inventory) {
    const statusEmoji =
      item.status === 'USED'
        ? '✅'
        : item.status === 'SHADOWED'
          ? '⚠️'
          : item.status === 'UNUSED'
            ? '❌'
            : '⚪';
    const issueText = item.issues.length > 0 ? item.issues.join('; ') : '-';
    lines.push(
      `| \`${item.path}\` | ${item.type} | \`${item.defaultValue}\` | ${statusEmoji} ${item.status} | ${issueText} |`,
    );
  }
  lines.push('');

  lines.push('## 三、问题详情 (Issues Detail)');
  lines.push('');

  const problemItems = result.inventory.filter(
    (item) => item.status === 'UNUSED' || item.status === 'SHADOWED',
  );

  if (problemItems.length === 0) {
    lines.push('无问题。');
  } else {
    for (const item of problemItems) {
      lines.push(`### ${item.path}`);
      lines.push('');
      lines.push(`- **文件**: ${item.file}`);
      lines.push(`- **状态**: ${item.status}`);
      lines.push(`- **问题**:`);
      if (item.status === 'UNUSED') {
        lines.push(`  - 配置项未被任何组件使用`);
      }
      for (const issue of item.issues) {
        lines.push(`  - ${issue}`);
      }
      lines.push('');
    }
  }

  lines.push('## 四、修复建议 (Fix Recommendations)');
  lines.push('');

  for (const item of problemItems) {
    if (item.status === 'UNUSED') {
      lines.push(`- **${item.path}**: 需要在相应组件中消费此配置，或从 schema/yml 中移除`);
    } else if (item.status === 'SHADOWED') {
      lines.push(`- **${item.path}**: 移除硬编码的 class，改为从配置读取并映射到 Tailwind class`);
    }
  }
  lines.push('');

  return lines.join('\n');
}

/**
 * Main audit function
 */
function runAudit(): AuditResult {
  console.log('🔍 Starting config effectiveness audit...\n');

  // Step 1: Extract config inventory
  console.log('📋 Step 1: Extracting config inventory...');
  const inventory = extractConfigInventory();
  console.log(`   Found ${inventory.length} config items\n`);

  // Step 2: Scan usage
  console.log('🔎 Step 2: Scanning config usage in source files...');
  scanConfigUsage(inventory);
  console.log('   Scan complete\n');

  // Step 3: Calculate summary
  const summary = {
    total: inventory.length,
    used: inventory.filter((i) => i.status === 'USED').length,
    readOnly: inventory.filter((i) => i.status === 'READ_ONLY').length,
    shadowed: inventory.filter((i) => i.status === 'SHADOWED').length,
    unused: inventory.filter((i) => i.status === 'UNUSED').length,
  };

  const result: AuditResult = {
    inventory,
    summary,
    exitCode: summary.shadowed + summary.unused > 0 ? 1 : 0,
  };

  // Step 4: Generate report
  console.log('📝 Step 3: Generating audit report...');
  const report = generateReport(result);
  const docsDir = path.join(process.cwd(), 'docs');
  if (!fs.existsSync(docsDir)) {
    fs.mkdirSync(docsDir, { recursive: true });
  }
  fs.writeFileSync(path.join(docsDir, 'config-audit.md'), report);
  console.log('   Report saved to docs/config-audit.md\n');

  // Print summary
  console.log('📊 Audit Summary:');
  console.log(`   Total: ${summary.total}`);
  console.log(`   ✅ Used: ${summary.used}`);
  console.log(`   ⚪ Read-only: ${summary.readOnly}`);
  console.log(`   ⚠️  Shadowed: ${summary.shadowed}`);
  console.log(`   ❌ Unused: ${summary.unused}\n`);

  if (result.exitCode !== 0) {
    console.error('❌ FAIL: Found config effectiveness issues!');
    console.error('   Please check docs/config-audit.md for details.\n');
  } else {
    console.log('✅ PASS: All configs are effective!\n');
  }

  return result;
}

// Run audit as main execution
const result = runAudit();
process.exit(result.exitCode);

export { runAudit, type ConfigItem, type AuditResult };
