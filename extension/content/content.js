(function () {
  if (typeof Readability === 'undefined') {
    return { error: 'Readability.js not loaded' };
  }

  const documentClone = document.cloneNode(true);

  try {
    const reader = new Readability(documentClone);
    const article = reader.parse();

    if (!article) {
      return {
        error: 'Could not extract content from this page',
        url: location.href,
        title: document.title
      };
    }

    // Extract image URLs from the article content HTML
    const images = [];
    const MIN_DIMENSION = 100;
    const MAX_IMAGES = 20;

    // Strategy 1: Parse Readability article HTML for images
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = article.content || '';
    const imgElements = tempDiv.querySelectorAll('img');
    console.log('[SiteGist] article.content img count:', imgElements.length);

    let imageIndex = 0;
    for (const img of imgElements) {
      if (imageIndex >= MAX_IMAGES) break;

      const src = img.getAttribute('src') || '';
      if (!src || src.startsWith('data:image/svg+xml')) continue;

      // Resolve relative URLs
      let fullSrc;
      try {
        fullSrc = new URL(src, location.href).href;
      } catch (_e) {
        continue;
      }

      // Use HTML attributes for dimensions (Readability preserves them)
      const width = parseInt(img.getAttribute('width'), 10) || 0;
      const height = parseInt(img.getAttribute('height'), 10) || 0;

      // Skip known small images if dimensions are available
      if (width > 0 && width < MIN_DIMENSION && height > 0 && height < MIN_DIMENSION) continue;
      if (width === 1 && height === 1) continue;

      images.push({
        base64: '',
        src: fullSrc,
        alt: img.getAttribute('alt') || '',
        width,
        height,
        index: imageIndex,
        needsFetch: true
      });
      console.log(`[SiteGist] image[${imageIndex}]:`, fullSrc.substring(0, 80), `${width}x${height}`);
      imageIndex++;
    }

    // Strategy 2: If Readability found no images, scan live DOM directly
    if (images.length === 0) {
      console.log('[SiteGist] No images from Readability, scanning live DOM...');
      const liveImgs = document.querySelectorAll('article img, [data-component="image"] img, .image_large img, figure img, .story-body img');
      console.log('[SiteGist] Live DOM candidate imgs:', liveImgs.length);
      for (const liveImg of liveImgs) {
        if (imageIndex >= MAX_IMAGES) break;
        const liveSrc = liveImg.currentSrc || liveImg.src || '';
        if (!liveSrc || liveSrc.startsWith('data:')) continue;
        const w = liveImg.naturalWidth || parseInt(liveImg.getAttribute('width'), 10) || 0;
        const h = liveImg.naturalHeight || parseInt(liveImg.getAttribute('height'), 10) || 0;
        if (w > 0 && w < MIN_DIMENSION && h > 0 && h < MIN_DIMENSION) continue;
        if (w === 1 && h === 1) continue;
        images.push({
          base64: '',
          src: liveSrc,
          alt: liveImg.getAttribute('alt') || '',
          width: w,
          height: h,
          index: imageIndex,
          needsFetch: true
        });
        console.log(`[SiteGist] live DOM image[${imageIndex}]:`, liveSrc.substring(0, 80), `${w}x${h}`);
        imageIndex++;
      }
    }

    console.log('[SiteGist] Total images extracted:', images.length);

    return {
      title: article.title || document.title,
      textContent: article.textContent || '',
      byline: article.byline || '',
      siteName: article.siteName || '',
      excerpt: article.excerpt || '',
      url: location.href,
      length: article.length || 0,
      images
    };
  } catch (e) {
    return {
      error: `Extraction failed: ${e.message}`,
      url: location.href,
      title: document.title
    };
  }
})();
