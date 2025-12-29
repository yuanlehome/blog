import { describe, expect, it } from 'vitest';
import {
  resolveAdapter,
  getAllAdapters,
  getAdapterById,
} from '../../scripts/import/adapters/index';

describe('Adapter Registry', () => {
  describe('resolveAdapter', () => {
    it('should resolve zhihu adapter for zhihu URLs', () => {
      const adapter = resolveAdapter('https://zhuanlan.zhihu.com/p/668888063');
      expect(adapter).not.toBeNull();
      expect(adapter?.id).toBe('zhihu');
      expect(adapter?.name).toBe('Zhihu Column');
    });

    it('should resolve medium adapter for medium URLs', () => {
      const adapter = resolveAdapter('https://medium.com/@user/article');
      expect(adapter).not.toBeNull();
      expect(adapter?.id).toBe('medium');
      expect(adapter?.name).toBe('Medium');
    });

    it('should resolve wechat adapter for wechat URLs', () => {
      const adapter = resolveAdapter('https://mp.weixin.qq.com/s/abc123');
      expect(adapter).not.toBeNull();
      expect(adapter?.id).toBe('wechat');
      expect(adapter?.name).toBe('WeChat');
    });

    it('should resolve others adapter for unknown URLs', () => {
      const adapter = resolveAdapter('https://example.com/article');
      expect(adapter).not.toBeNull();
      expect(adapter?.id).toBe('others');
      expect(adapter?.name).toBe('Others (Generic)');
    });

    it('should resolve specific adapters before others', () => {
      // Test that specific adapters have priority
      const zhihuAdapter = resolveAdapter('https://zhuanlan.zhihu.com/p/123');
      expect(zhihuAdapter?.id).toBe('zhihu');

      const randomAdapter = resolveAdapter('https://random.com/article');
      expect(randomAdapter?.id).toBe('others');
    });
  });

  describe('getAllAdapters', () => {
    it('should return all registered adapters', () => {
      const adapters = getAllAdapters();
      expect(adapters.length).toBeGreaterThanOrEqual(4);

      const ids = adapters.map((a) => a.id);
      expect(ids).toContain('zhihu');
      expect(ids).toContain('medium');
      expect(ids).toContain('wechat');
      expect(ids).toContain('others');
    });

    it('should have others adapter as last priority', () => {
      const adapters = getAllAdapters();
      const lastAdapter = adapters[adapters.length - 1];
      expect(lastAdapter.id).toBe('others');
    });
  });

  describe('getAdapterById', () => {
    it('should return adapter by id', () => {
      const zhihu = getAdapterById('zhihu');
      expect(zhihu).not.toBeNull();
      expect(zhihu?.id).toBe('zhihu');

      const medium = getAdapterById('medium');
      expect(medium).not.toBeNull();
      expect(medium?.id).toBe('medium');

      const wechat = getAdapterById('wechat');
      expect(wechat).not.toBeNull();
      expect(wechat?.id).toBe('wechat');

      const others = getAdapterById('others');
      expect(others).not.toBeNull();
      expect(others?.id).toBe('others');
    });

    it('should return null for non-existent adapter', () => {
      const adapter = getAdapterById('non-existent');
      expect(adapter).toBeNull();
    });
  });

  describe('Adapter canHandle method', () => {
    it('zhihu adapter should only handle zhihu column URLs', () => {
      const zhihu = getAdapterById('zhihu');
      expect(zhihu?.canHandle('https://zhuanlan.zhihu.com/p/123')).toBe(true);
      expect(zhihu?.canHandle('https://www.zhihu.com/question/123')).toBe(false); // Not a column
      expect(zhihu?.canHandle('https://medium.com/article')).toBe(false);
      expect(zhihu?.canHandle('https://mp.weixin.qq.com/s/abc')).toBe(false);
    });

    it('medium adapter should only handle medium URLs', () => {
      const medium = getAdapterById('medium');
      expect(medium?.canHandle('https://medium.com/@user/article')).toBe(true);
      expect(medium?.canHandle('https://subdomain.medium.com/article')).toBe(true);
      expect(medium?.canHandle('https://zhuanlan.zhihu.com/p/123')).toBe(false);
      expect(medium?.canHandle('https://mp.weixin.qq.com/s/abc')).toBe(false);
    });

    it('wechat adapter should only handle wechat URLs', () => {
      const wechat = getAdapterById('wechat');
      expect(wechat?.canHandle('https://mp.weixin.qq.com/s/abc123')).toBe(true);
      expect(wechat?.canHandle('https://zhuanlan.zhihu.com/p/123')).toBe(false);
      expect(wechat?.canHandle('https://medium.com/article')).toBe(false);
    });

    it('others adapter should handle all URLs', () => {
      const others = getAdapterById('others');
      expect(others?.canHandle('https://example.com/article')).toBe(true);
      expect(others?.canHandle('https://any-site.com/page')).toBe(true);
      expect(others?.canHandle('https://zhuanlan.zhihu.com/p/123')).toBe(true);
    });
  });

  describe('Adapter interface', () => {
    it('all adapters should have required properties', () => {
      const adapters = getAllAdapters();

      for (const adapter of adapters) {
        expect(adapter).toHaveProperty('id');
        expect(adapter).toHaveProperty('name');
        expect(adapter).toHaveProperty('canHandle');
        expect(adapter).toHaveProperty('fetchArticle');
        expect(typeof adapter.canHandle).toBe('function');
        expect(typeof adapter.fetchArticle).toBe('function');
        expect(['zhihu', 'medium', 'wechat', 'others']).toContain(adapter.id);
      }
    });
  });
});
