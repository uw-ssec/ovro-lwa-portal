// Initialize Mermaid
document$.subscribe(() => {
  if (typeof mermaid !== 'undefined') {
    mermaid.initialize({
      startOnLoad: true,
      theme: 'default',
      securityLevel: 'loose',
      fontFamily: 'inherit'
    });
    mermaid.run();
  }
});
