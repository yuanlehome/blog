// Client-side post search within a tag page
(function () {
  const postsDataEl = document.getElementById('posts-data');
  if (!postsDataEl) return;

  const postsData = JSON.parse(postsDataEl.textContent || '[]');
  const searchInput = document.getElementById('post-search') as HTMLInputElement | null;
  const postsContainer = document.getElementById('posts-container');

  if (!searchInput || !postsContainer) return;

  const originalHTML = postsContainer.innerHTML;

  function filterPosts() {
    const searchTerm = searchInput!.value.toLowerCase().trim();

    if (!searchTerm) {
      postsContainer!.innerHTML = originalHTML;
      return;
    }

    const filtered = postsData.filter((post: any) => post.title.toLowerCase().includes(searchTerm));

    if (filtered.length === 0) {
      postsContainer!.innerHTML =
        '<p class="text-zinc-500 dark:text-zinc-400">未找到匹配的文章</p>';
      return;
    }

    const postElements = postsContainer!.querySelectorAll('#post-list > li');
    postElements.forEach((el) => {
      const postLink = el.querySelector('a');
      if (!postLink) return;

      const href = postLink.getAttribute('href');
      const slug = href?.split('/').filter(Boolean).pop();
      const isVisible = filtered.some((p: any) => p.slug === slug);

      (el as HTMLElement).style.display = isVisible ? '' : 'none';
    });
  }

  searchInput.addEventListener('input', filterPosts);
})();
