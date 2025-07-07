// rotulagem.js — script de rotulagem para ser chamado sob demanda
(function(){
  // Cria storage global (se não existir)
  window.__ux_examples = window.__ux_examples || [];
  window.__ux_training = true;

  // Headings acima do elemento (array de até 5 textos)
  function getHeadingsAboveArray(el, max = 5) {
      var results = [];
      var used = {};
      var stopAtTags = ['body', 'html'];
      var tags = ['h1','h2','h3','h4','h5','h6','p'];
      var node = el;
      while (node && results.length < max && stopAtTags.indexOf((node.tagName||'').toLowerCase()) === -1) {
          var sib = node.previousElementSibling;
          while (sib && results.length < max) {
              var tag = (sib.tagName||'').toLowerCase();
              if (tags.indexOf(tag) > -1) {
                  var txt = (sib.innerText||'').replace(/\s+/g, ' ').trim();
                  if (txt.length > 2 && !used[txt]) {
                      results.push(txt);
                      used[txt] = 1;
                  }
              }
              sib = sib.previousElementSibling;
          }
          var tagSelf = (node.tagName||'').toLowerCase();
          if (tags.indexOf(tagSelf) > -1) {
              var txt = (node.innerText||'').replace(/\s+/g, ' ').trim();
              if (txt.length > 2 && !used[txt]) {
                  results.push(txt);
                  used[txt] = 1;
              }
          }
          node = node.parentElement;
      }
      return results;
  }

  // Posição vertical em percentual
  function getYPercent(el) {
      var rect = el.getBoundingClientRect();
      var scrollY = window.scrollY || window.pageYOffset;
      var yAbs = rect.top + scrollY;
      var docHeight = Math.max(document.documentElement.scrollHeight, document.body.scrollHeight);
      return Math.round((yAbs / docHeight) * 100);
  }

  // Índice do elemento entre os irmãos
  function getSiblingIndex(el) {
      var i = 0;
      while ((el = el.previousElementSibling) != null) i++;
      return i;
  }

  // Painel visual de sugestão/rotulação
  function showSessionPanel(options, posYPercent, callback) {
      var old = document.getElementById('__ux_session_panel');
      if(old) old.parentNode.removeChild(old);
      var oldBack = document.getElementById('__ux_session_backdrop');
      if(oldBack) oldBack.parentNode.removeChild(oldBack);

      var backdrop = document.createElement('div');
      backdrop.id = '__ux_session_backdrop';
      backdrop.style.position = 'fixed';
      backdrop.style.top = 0;
      backdrop.style.left = 0;
      backdrop.style.width = '100vw';
      backdrop.style.height = '100vh';
      backdrop.style.zIndex = 99998;
      backdrop.style.background = 'rgba(0,0,0,0.45)';
      document.body.appendChild(backdrop);

      var div = document.createElement('div');
      div.id = '__ux_session_panel';
      div.style.position = 'fixed';
      div.style.top = '50%';
      div.style.left = '50%';
      div.style.transform = 'translate(-50%, -50%)';
      div.style.background = '#fff';
      div.style.border = '2px solid #333';
      div.style.zIndex = 99999;
      div.style.padding = '22px 28px 20px 28px';
      div.style.borderRadius = '14px';
      div.style.boxShadow = '0 8px 32px #0008';
      div.style.fontFamily = 'sans-serif';
      div.style.minWidth = '340px';
      div.style.maxWidth = '96vw';

      var html = "<b style='font-size:1.1em'>Qual a sessão deste elemento?</b>";
      html += `<div style='font-size:0.96em;margin-bottom:7px;color:#666'>Posição Y: <b>${posYPercent}%</b></div>`;
      html += "<form id='__ux_form' autocomplete='off' style='margin-top:8px'>";
      options.forEach(function(opt, idx){
          html += `<label style="margin:7px 0;display:block;font-size:1.01em;line-height:1.4;">
          <input type="radio" name="sessao" value="${opt}" style="vertical-align:middle;margin-right:7px" ${(idx===0?'checked':'')}/> ${opt}</label>`;
      });
      html += `<label style="margin:10px 0 6px 0;display:block;font-size:1.01em;line-height:1.4;">
          <input type="radio" name="sessao" value="Outra" style="vertical-align:middle;margin-right:7px"/> Outra
          <input type="text" id="__ux_outro_text" placeholder="Digite a sessão" style="width:78%;font-size:1em;padding:2px 8px;margin-left:8px;display:none" autocomplete="off"/>
          </label>`;
      html += '<button style="margin-top:14px;padding:5px 24px;font-size:1.07em;background:#149C3B;color:#fff;border:none;border-radius:8px;cursor:pointer" type="submit">Salvar</button>';
      html += '</form>';
      div.innerHTML = html;
      document.body.appendChild(div);

      document.body.style.overflow = 'hidden';

      // Input “Outra” só aparece ao marcar
      var radios = div.querySelectorAll('input[type="radio"]');
      var outroInput = div.querySelector('#__ux_outro_text');
      radios.forEach(function(r){
          r.addEventListener('change', function(){
              if(this.value === 'Outra') {
                  outroInput.style.display = '';
                  setTimeout(() => outroInput.focus(), 80);
              }
              else outroInput.style.display = 'none';
          });
      });

      div.querySelector('form').onsubmit = function(e){
          e.preventDefault();
          var value = div.querySelector('input[type="radio"]:checked').value;
          if(value === 'Outra'){
              var txt = outroInput.value.trim();
              if(txt.length < 2){
                  alert("Digite um nome para a sessão");
                  outroInput.focus();
                  return;
              }
              value = txt;
          }
          div.remove();
          backdrop.remove();
          document.body.style.overflow = '';
          callback(value);
      }
  }

  // Nova versão: agora coleta headings, Y%, índice do irmão e tudo textual
  function extractFeatures(el) {
      function safeStr(val) {
          if (!val) return '';
          if (typeof val === 'string') return val;
          if (typeof val === 'object') return Array.from(val).join(' ');
          return '';
      }
      var parents = [];
      var p = el.parentElement;
      for (var i = 0; i < 3 && p; i++) {
          parents.push({
              tag: p.tagName,
              class: safeStr(p.className),
              id: p.id,
              text: (p.innerText || '').trim()
          });
          p = p.parentElement;
      }
      return {
          tag: el.tagName,
          class: safeStr(el.className),
          id: el.id,
          text: (el.innerText || el.value || '').trim(),
          parents: parents,
          length: (el.innerText || '').length,
          y: getYPercent(el),
          siblingIndex: getSiblingIndex(el),
          contextHeadings: getHeadingsAboveArray(el, 5)
      };
  }
  window.extractFeatures = extractFeatures;

  // Handler: ignora cliques no próprio painel/modal!
  window.__ux_clickListener = function(e) {
      var panel = document.getElementById('__ux_session_panel');
      var backdrop = document.getElementById('__ux_session_backdrop');
      if ((panel && panel.contains(e.target)) || (backdrop && backdrop.contains(e.target))) {
          return;
      }
      if (!window.__ux_training) return;
      e.preventDefault();
      var features = extractFeatures(e.target);
      var yPercent = features.y;
      var options = features.contextHeadings.slice(0, 10);
      var defaults = ['Menu','Hero','Conteúdo','Rodape','Footer','Header','Topo'];
      for (var i=0; options.length < 10 && i<defaults.length; i++) {
          if (options.indexOf(defaults[i]) === -1) options.push(defaults[i]);
      }
      showSessionPanel(options, yPercent, function(sessao){
          window.__ux_examples.push(Object.assign({sessao: sessao}, features));
          console.log("[AUTO-UX] Exemplo salvo! Total de exemplos:", window.__ux_examples.length);
      });
  };

  document.removeEventListener('click', window.__ux_clickListener, true); // Garante que não duplica
  document.addEventListener('click', window.__ux_clickListener, true);

  // Opcional: exportar para JSON
  window.exportarExemplosUX = function(){
      var a = document.createElement('a');
      a.href = URL.createObjectURL(new Blob([JSON.stringify(window.__ux_examples, null, 2)], {type:'application/json'}));
      a.download = "ux_examples.json";
      a.click();
  };

  console.log(">>> [AUTO-UX] Script de rotulagem carregado! Clique nos elementos para rotular. Exporte exemplos com exportarExemplosUX().");

})();
