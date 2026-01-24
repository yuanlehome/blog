// Client-side tag filtering and sorting for tags index page
(function () {
  const allTagsDataEl = document.getElementById('tags-data');
  if (!allTagsDataEl) return;

  const allTagsData = JSON.parse(allTagsDataEl.textContent || '[]');
  const searchInput = document.getElementById('tag-search') as HTMLInputElement | null;
  const sortSelect = document.getElementById('tag-sort') as HTMLSelectElement | null;
  const container = document.getElementById('tags-container');

  if (!searchInput || !sortSelect || !container) return;

  function renderTags(tags: any[]) {
    if (tags.length === 0) {
      container!.innerHTML = '<p class="text-zinc-500 dark:text-zinc-400">未找到匹配的标签</p>';
      return;
    }

    const base = window.location.pathname.replace(/\/tags\/$/, '');
    container!.innerHTML = tags
      .map(
        (tag) => `
        <a
          href="${base}/tags/${tag.slug}/"
          class="tag-chip inline-flex items-center gap-1.5 rounded-full border transition-colors duration-200 font-medium text-sm px-3 py-1.5 cursor-pointer
            border-sky-200 bg-sky-50 text-sky-700 hover:bg-sky-100 hover:border-sky-300 
            dark:border-sky-900 dark:bg-sky-950 dark:text-sky-300 dark:hover:bg-sky-900 dark:hover:border-sky-700
            focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500 focus-visible:ring-offset-2 dark:focus-visible:ring-offset-zinc-900"
          data-tag-slug="${tag.slug}"
          data-testid="tag-${tag.slug}"
        >
          <span class="tag-name pointer-events-none">${tag.name}</span>
          <span class="tag-count inline-flex items-center justify-center min-w-[1.25rem] h-5 px-1.5 rounded-full text-[0.7rem] font-semibold bg-sky-200/50 text-sky-800 dark:bg-sky-800/50 dark:text-sky-200 pointer-events-none">
            ${tag.count}
          </span>
        </a>
      `,
      )
      .join('');
  }

  function filterAndSort() {
    const searchTerm = searchInput!.value.toLowerCase().trim();
    const sortBy = sortSelect!.value;

    let filtered = allTagsData.filter((tag: any) => tag.name.toLowerCase().includes(searchTerm));

    switch (sortBy) {
      case 'name':
        filtered.sort((a: any, b: any) => a.name.localeCompare(b.name));
        break;
      case 'recent':
        filtered.sort(
          (a: any, b: any) => new Date(b.latestDate).getTime() - new Date(a.latestDate).getTime(),
        );
        break;
      case 'count':
      default:
        filtered.sort((a: any, b: any) => {
          if (b.count !== a.count) return b.count - a.count;
          return a.name.localeCompare(b.name);
        });
    }

    renderTags(filtered);
  }

  searchInput.addEventListener('input', filterAndSort);
  sortSelect.addEventListener('change', filterAndSort);
})();
