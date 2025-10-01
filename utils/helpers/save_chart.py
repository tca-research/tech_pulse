from IPython.display import HTML

def save_chart(chunk_id=None):
    if chunk_id is None:
        # Auto-detect version (Option 2 from before)
        html = """
        <div class="save-wrapper" style="width:100%; display:flex;">
          <div style="margin-left:auto; padding:10px; border-radius:6px; z-index: 10">
            <div class="dropdown"> 
              <button class="btn btn-outline-primary dropdown-toggle minimal-arrow-btn save-btn"
                      type="button"
                      data-bs-toggle="dropdown"
                      aria-expanded="false"> 
                Save ▼ 
              </button> 
              <ul class="dropdown-menu"></ul>
            </div>
          </div>
        </div>
        <script>
        (function() {
          const wrapper = document.currentScript.previousElementSibling.closest('.save-wrapper');
          const parent = wrapper.closest('div[id^="cell-"]');
          if (parent) {
            const cid = parent.id.replace(/^cell-/, '');
            const menu = wrapper.querySelector('.dropdown-menu');
            menu.innerHTML = `
              <li><a class="dropdown-item" href="${cid}.csv" download>Download CSV</a></li>
              <li><a class="dropdown-item" href="${cid}.png" download>Download PNG</a></li>
            `;
          }
        })();
        </script>
        """
    else:
        # Explicit ID version
        html = f"""
        <div style="width:100%; display:flex;">
          <div style="margin-left:auto; padding:10px; border-radius:6px; z-index: 10">
            <div class="dropdown"> 
              <button class="btn btn-outline-primary dropdown-toggle minimal-arrow-btn" 
                      type="button" 
                      id="{chunk_id}-download-menu" 
                      data-bs-toggle="dropdown" 
                      aria-expanded="false"> 
                Save ▼ 
              </button> 
              <ul class="dropdown-menu" aria-labelledby="{chunk_id}-download-menu"> 
                <li><a class="dropdown-item" href="{chunk_id}.csv" download>Download CSV</a></li> 
                <li><a class="dropdown-item" href="{chunk_id}.png" download>Download PNG</a></li> 
              </ul> 
            </div>
          </div>
        </div>
        """
    return HTML(html)
