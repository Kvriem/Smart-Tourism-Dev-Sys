// Enhanced Modal Event Handlers to Fix Page Shrinking and Scroll-to-Top Issues
// Specifically designed for Dash Bootstrap Components modals

(function() {
    'use strict';
    
    let scrollPosition = 0;
    let isModalOpen = false;
    let bodyOriginalStyle = '';
    
    // Store original body styles
    function storeOriginalBodyStyle() {
        bodyOriginalStyle = {
            position: document.body.style.position,
            top: document.body.style.top,
            width: document.body.style.width,
            overflow: document.body.style.overflow,
            paddingRight: document.body.style.paddingRight
        };
    }
    
    // Function to handle modal opening
    function handleModalOpen() {
        if (isModalOpen) return; // Prevent multiple calls
        
        console.log('Modal opening - preserving scroll position');
        isModalOpen = true;
        
        // Store current scroll position
        scrollPosition = window.pageYOffset || document.documentElement.scrollTop;
        
        // Store original body styles
        storeOriginalBodyStyle();
        
        // Apply styles to prevent body scroll and page shifting
        document.body.style.position = 'fixed';
        document.body.style.top = `-${scrollPosition}px`;
        document.body.style.width = '100%';
        document.body.style.overflow = 'hidden';
        document.body.style.paddingRight = '0px'; // Prevent Bootstrap padding-right
        
        // Add class to prevent any additional styling issues
        document.body.classList.add('modal-scroll-lock');
    }
    
    // Function to handle modal closing
    function handleModalClose() {
        if (!isModalOpen) return; // Prevent multiple calls
        
        console.log('Modal closing - restoring scroll position');
        isModalOpen = false;
        
        // Remove the scroll lock class
        document.body.classList.remove('modal-scroll-lock');
        
        // Restore original body styles
        document.body.style.position = bodyOriginalStyle.position || '';
        document.body.style.top = bodyOriginalStyle.top || '';
        document.body.style.width = bodyOriginalStyle.width || '';
        document.body.style.overflow = bodyOriginalStyle.overflow || '';
        document.body.style.paddingRight = bodyOriginalStyle.paddingRight || '';
        
        // Use requestAnimationFrame to ensure DOM is ready before scrolling
        requestAnimationFrame(() => {
            window.scrollTo(0, scrollPosition);
            console.log(`Restored scroll position to: ${scrollPosition}`);
        });
    }
      // Check if any modal is currently visible
    function isAnyModalVisible() {
        const modals = document.querySelectorAll('#word-analysis-modal, #nationality-analysis-modal');
        return Array.from(modals).some(modal => {
            // Check multiple conditions for Dash Bootstrap Components
            const hasShowClass = modal.classList.contains('show');
            const isDisplayed = modal.style.display !== 'none' && modal.style.display !== '';
            const hasVisibility = modal.style.visibility !== 'hidden';
            const computedStyle = window.getComputedStyle(modal);
            const isVisible = computedStyle.display !== 'none' && computedStyle.visibility !== 'hidden';
            
            return hasShowClass || (isDisplayed && hasVisibility && isVisible);
        });
    }
    
    // Enhanced modal state detection
    function checkModalState() {
        const anyModalVisible = isAnyModalVisible();
        
        if (anyModalVisible && !isModalOpen) {
            handleModalOpen();
        } else if (!anyModalVisible && isModalOpen) {
            handleModalClose();
        }
    }
    
    // Initialize when DOM is ready
    function initialize() {
        console.log('Initializing modal scroll fix');
        
        // Set up mutation observers for each modal
        const modalSelectors = ['#word-analysis-modal', '#nationality-analysis-modal'];
        
        modalSelectors.forEach(selector => {
            const modalElement = document.querySelector(selector);
            if (modalElement) {
                console.log(`Setting up observer for ${selector}`);
                
                // Create observer for style and class changes
                const observer = new MutationObserver((mutations) => {
                    let shouldCheck = false;
                    mutations.forEach((mutation) => {
                        if (mutation.type === 'attributes' && 
                            (mutation.attributeName === 'style' || mutation.attributeName === 'class')) {
                            shouldCheck = true;
                        }
                    });
                    
                    if (shouldCheck) {
                        // Debounce the check
                        setTimeout(checkModalState, 10);
                    }
                });
                
                observer.observe(modalElement, {
                    attributes: true,
                    attributeFilter: ['style', 'class']
                });
                
                // Also listen for traditional Bootstrap events if available
                if (modalElement.addEventListener) {
                    modalElement.addEventListener('show.bs.modal', handleModalOpen);
                    modalElement.addEventListener('hidden.bs.modal', handleModalClose);
                }
            }
        });
        
        // Backup polling mechanism
        setInterval(checkModalState, 100);
        
        // Watch for body style changes that might interfere
        const bodyObserver = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'attributes' && mutation.attributeName === 'style') {
                    const body = mutation.target;
                    
                    // If no modal is open but body has modal-open styles, clean them up
                    if (!isAnyModalVisible() && !isModalOpen) {
                        if (body.style.overflow === 'hidden' || body.style.paddingRight) {
                            setTimeout(() => {
                                if (!isAnyModalVisible()) {
                                    body.style.overflow = '';
                                    body.style.paddingRight = '';
                                    body.classList.remove('modal-open');
                                }
                            }, 100);
                        }
                    }
                }
            });
        });
        
        bodyObserver.observe(document.body, {
            attributes: true,
            attributeFilter: ['style', 'class']
        });
    }
    
    // Initialize when DOM is loaded
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initialize);
    } else {
        initialize();
    }
    
    // Also initialize after a short delay to catch any late-loading elements
    setTimeout(initialize, 1000);
    
})();