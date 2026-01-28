/**
 * Tests for config audit script
 */

import { describe, it, expect, beforeAll } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';

describe('Config Audit Script', () => {
  const auditReportPath = path.join(process.cwd(), 'docs', 'config-audit.md');

  beforeAll(() => {
    // Ensure the audit report exists (run audit before tests if needed)
    if (!fs.existsSync(auditReportPath)) {
      throw new Error('Audit report not found. Run "npm run config:audit" before running tests.');
    }
  });

  describe('Audit Report', () => {
    it('should generate config-audit.md in docs folder', () => {
      expect(fs.existsSync(auditReportPath)).toBe(true);
    });

    it('should contain audit summary section', () => {
      const content = fs.readFileSync(auditReportPath, 'utf-8');
      expect(content).toContain('## 一、审计摘要');
      expect(content).toContain('总配置项');
      expect(content).toContain('已生效');
    });

    it('should contain config inventory section', () => {
      const content = fs.readFileSync(auditReportPath, 'utf-8');
      expect(content).toContain('## 二、配置项清单');
      expect(content).toContain('配置路径');
      expect(content).toContain('类型');
      expect(content).toContain('状态');
    });

    it('should contain prevention strategy section', () => {
      const content = fs.readFileSync(auditReportPath, 'utf-8');
      expect(content).toContain('防止配置失效的措施');
      expect(content).toContain('自动化审计');
      expect(content).toContain('npm run config:audit');
    });

    it('should list alignment configs as USED', () => {
      const content = fs.readFileSync(auditReportPath, 'utf-8');
      expect(content).toContain('layout.alignment.headerAlign');
      expect(content).toContain('layout.alignment.footerAlign');
      expect(content).toContain('layout.alignment.postMetaAlign');
      // All three should be marked as USED
      expect(content).toMatch(/headerAlign.*✅ USED/);
      expect(content).toMatch(/footerAlign.*✅ USED/);
      expect(content).toMatch(/postMetaAlign.*✅ USED/);
    });

    it('should not show any UNUSED configs', () => {
      const content = fs.readFileSync(auditReportPath, 'utf-8');
      // Check summary doesn't show unused configs
      expect(content).toMatch(/完全未使用.*0/);
    });

    it('should not show any SHADOWED configs', () => {
      const content = fs.readFileSync(auditReportPath, 'utf-8');
      // Check summary doesn't show shadowed configs
      expect(content).toMatch(/被硬编码覆盖.*0/);
    });
  });

  describe('Config Effectiveness', () => {
    it('should verify alignment utility functions exist', () => {
      const alignmentUtilPath = path.join(process.cwd(), 'src/lib/ui/alignment.ts');
      expect(fs.existsSync(alignmentUtilPath)).toBe(true);
      const content = fs.readFileSync(alignmentUtilPath, 'utf-8');
      expect(content).toContain('alignToTextClass');
      expect(content).toContain('alignToJustifyClass');
      expect(content).toContain('alignToItemsClass');
    });

    it('should verify Header uses alignment config', () => {
      const headerPath = path.join(process.cwd(), 'src/components/Header.astro');
      const content = fs.readFileSync(headerPath, 'utf-8');
      expect(content).toContain('getLayoutConfig');
      expect(content).toContain('alignTo');
      expect(content).toContain('headerAlign');
    });

    it('should verify Footer uses alignment config', () => {
      const footerPath = path.join(process.cwd(), 'src/components/Footer.astro');
      const content = fs.readFileSync(footerPath, 'utf-8');
      expect(content).toContain('getLayoutConfig');
      expect(content).toContain('alignToTextClass');
      expect(content).toContain('footerAlign');
    });

    it('should verify post page uses postMetaAlign config', () => {
      const postPagePath = path.join(process.cwd(), 'src/pages/[...slug].astro');
      const content = fs.readFileSync(postPagePath, 'utf-8');
      expect(content).toContain('getLayoutConfig');
      expect(content).toContain('alignTo');
      expect(content).toContain('postMetaAlign');
    });
  });

  describe('Audit Script Execution', () => {
    it('should have config:audit npm script', () => {
      const packageJsonPath = path.join(process.cwd(), 'package.json');
      const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf-8'));
      expect(packageJson.scripts).toHaveProperty('config:audit');
      expect(packageJson.scripts['config:audit']).toContain('config-audit.ts');
    });

    it('should have audit script file', () => {
      const scriptPath = path.join(process.cwd(), 'scripts/config-audit.ts');
      expect(fs.existsSync(scriptPath)).toBe(true);
    });
  });
});
