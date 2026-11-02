(() => {
  const lessons = {
    embedding: {
      title:'Embedding：沿词表切开查表',role:'位于模型入口。将 token ID 查成完整的隐藏向量，再交给 Decoder。',axis:'切分轴 / 词表行',
      steps:['切分前','权重切开','收集 ID','本地查表','求和归还'],
      explanations:[
        ['各 DP 保存完整词表','DP0 为 A 查 id 1，得到 [0, 1]；DP1 为 B 查 id 3，得到 [2, 1]。每个 die 都保存 4 行词表权重。','LOCAL LOOKUP'],
        ['每个 die 常驻半个词表','DP0 保存 id 0–1，DP1 保存 id 2–3。每行的隐藏维度仍为 2；权重在模型加载时完成切分。','WEIGHT SHARD · VOCAB / 2'],
        ['把两个请求的 ID 收到组内','AllGather 收集 A 的 id 1、B 的 id 3。两个 die 都收到这两个 ID，分别查各自保存的词表分片。','ALLGATHER · TOKEN ID'],
        ['命中的分片给向量，其余给零','DP0 对 A 给 [0, 1]，对 B 给 [0, 0]；DP1 对 A 给 [0, 0]，对 B 给 [2, 1]。同一 ID 只有所属分片提供有效值。','LOCAL LOOKUP + MASK'],
        ['按请求求和并归还','ReduceScatter 对同一 token 的向量求和，再分回原 DP：A 的 [0, 1] 回 DP0，B 的 [2, 1] 回 DP1。','REDUCESCATTER · SUM']
      ],
      takeaway:'完整词表 4×2 → 每 die 的词表分片 2×2。交换 token ID、汇总查表结果，输出向量仍是完整 2 维。',
      config:'--data-parallel-size 8 \\\n--tensor-parallel-size 2 \\\n--additional-config \'{"finegrained_tp_config":{"embedding_tensor_parallel_size":4}}\'',
      note:'16 die 示例：DP8 × TP2、Embedding=4，词表按 4 份分到 DP 域。DP 是切分份数的整数倍；主模型与 MTP 共享 Embedding 分片。PCP=1。',
      root:'#embedding-dp-steps',prev:'#embed-prev',next:'#embed-next'
    },
    oproj:{
      title:'OProj：沿输入特征切投影',role:'位于每层 Attention 末尾。将 Attention 的输出投影回隐藏维度，接入后续残差与 FFN。',axis:'切分轴 / 输入特征 K',
      steps:['切分前','权重切开','交换输入','分片计算','求和归还'],
      explanations:[
        ['各自完成一次完整投影','A=[1,2,3,4] 在 DP0，B=[5,6,7,8] 在 DP1。两边都保存 4×2 权重，分别得到 A=[12,5]、B=[28,13]。','LOCAL MATMUL'],
        ['权重按输入维度分成两片','DP0 保存对应输入维 1–2 的 2×2 权重，DP1 保存输入维 3–4 的 2×2 权重。每片仍覆盖全部输出维。','WEIGHT SHARD · K / 2'],
        ['把输入特征送到对应权重片','AllToAll：A 后两维 [3,4] 发到 DP1，B 前两维 [5,6] 发到 DP0。DP0 汇集 A、B 的前片，DP1 汇集后片。','ALLTOALL · INPUT FEATURES'],
        ['各片计算全输出维度的贡献','DP0 算出 A 的 [1,2]、B 的 [5,6]；DP1 算出 A 的 [11,3]、B 的 [23,7]。两片贡献相加才是完整投影。','LOCAL MATMUL · PARTIAL SUM'],
        ['同一 token 的两片贡献相加','ReduceScatter：A 得到 [1,2]+[11,3]=[12,5]，回 DP0；B 得到 [5,6]+[23,7]=[28,13]，回 DP1。','REDUCESCATTER · SUM']
      ],
      takeaway:'完整投影权重 4×2 → 每 die 权重 2×2。每个 die 分担输入特征对应的计算，再对同一 token 的部分贡献求和。',
      config:'--data-parallel-size 16 \\\n--tensor-parallel-size 1 \\\n--compilation-config \'{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":384}\' \\\n--additional-config \'{"finegrained_tp_config":{"oproj_tensor_parallel_size":4},"scheduler_config":{"recompute_scheduler_enable":true}}\'',
      note:'16 die 示例：DP16 × TP1、OProj=4、PCP=1。D 侧图捕获容量覆盖最大解码步；max-num-seqs=64、MTP=5 时容量为 384。P 侧使用行数对齐适配。',
      root:'#oproj-dp-explainer',prev:'[data-op-prev]',next:'[data-op-next]'
    },
    mlp:{
      title:'MLP：沿中间通道成对切分',role:'位于 Decoder 的 FFN 位置。GLM5.3Flash 第 0–2 层使用 dense MLP；第 3–44 层为 MoE，使用 EP。',axis:'切分轴 / 中间通道',
      steps:['切分前','权重切开','收集输入','分片计算','求和归还'],
      explanations:[
        ['各 DP 保存完整 dense MLP','每个输入隐藏向量有 2 维。gate、up 各投影到 4 个中间通道，经 SwiGLU 后，由 down 投影回 2 维。','GATE / UP → SWIGLU → DOWN'],
        ['同一通道的 gate / up / down 一起切','DP0 负责中间通道 1–2，DP1 负责 3–4。gate 与 up 的输出通道对应保留，down 保留对应输入通道及全部隐藏输出维。','WEIGHT SHARD · INTERMEDIATE / 2'],
        ['每片权重收到完整的隐藏输入','AllGather 按 token 收集 A、B 的输入。两边都拿到每个 token 的完整隐藏向量 [a1,a2] / [b1,b2]。','ALLGATHER · HIDDEN STATES'],
        ['各自算中间通道，再投影成部分和','各 die 只算自己的 gate/up 通道和 SwiGLU。对应 down 分片将这些通道投影到完整 2 维，形成前片贡献 p 和后片贡献 q。','LOCAL GATE / UP → SWIGLU → DOWN'],
        ['按同一 token 的贡献求和','ReduceScatter 对 A 的 pA+qA 求和后交回 DP0，对 B 的 pB+qB 求和后交回 DP1，得到完整隐藏向量。','REDUCESCATTER · SUM']
      ],
      takeaway:'每 die 的中间通道 4 → 2，gate/up 与 down 配对分片。最终汇总的是各通道对完整隐藏输出的贡献。',
      config:'--data-parallel-size 16 \\\n--tensor-parallel-size 1 \\\n--compilation-config \'{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":384}\' \\\n--additional-config \'{"finegrained_tp_config":{"mlp_tensor_parallel_size":4},"scheduler_config":{"recompute_scheduler_enable":true}}\'',
      note:'16 die 示例：DP16 × TP1、MLP=4、PCP=1。D 侧使用 recompute 调度与解码图，捕获容量 384 对应 max-num-seqs=64、MTP=5。P 侧在 gate/up 入口对齐 token 行。',
      root:'#mlp-dp-steps',prev:'[data-prev]',next:'[data-next]'
    },
    lmhead:{
      title:'LM Head：沿词表分担打分',role:'位于模型出口、最终 RMSNorm 之后。将隐藏向量投影为整个词表的 logits，交给采样器选下一个 token。',axis:'切分轴 / 词表输出',
      steps:['切分前','权重切开','交换并计算','分数归还'],
      explanations:[
        ['各 DP 为自己的请求算全词表','DP0 为 A 计算词 1–4 的分数 [2,1,4,0]；DP1 为 B 计算 [1,3,0,2]。每个 die 保存完整词表的 Head 权重。','LOCAL MATMUL · FULL VOCAB'],
        ['每个 die 保存一半词表权重','DP0 保存词 1–2 的权重，DP1 保存词 3–4。每个词对应的权重仍接收完整隐藏向量。','WEIGHT SHARD · VOCAB / 2'],
        ['收集隐藏向量，分别算半个词表','AllGather 让两个 die 都收到 A、B 的完整隐藏向量。DP0 为两个请求算词 1–2，DP1 为两个请求算词 3–4。','ALLGATHER → LOCAL MATMUL'],
        ['把同一请求的词表分数拼回去','AllToAll 按请求重排：A 的后半词表分数送回 DP0，B 的前半词表分数送回 DP1。A、B 各自收齐全词表 logits 后采样。','ALLTOALL · VOCAB LOGITS']
      ],
      takeaway:'完整词表权重 → 每 die 半个词表权重。两边都处理 A、B 的隐藏向量，各算半个词表，再把同一请求的词表分数拼齐。',
      config:'--data-parallel-size 8 \\\n--tensor-parallel-size 2 \\\n--no-enable-batch-sharded-sampling \\\n--additional-config \'{"finegrained_tp_config":{"lmhead_tensor_parallel_size":4}}\'',
      note:'16 die 示例：DP8 × TP2、LM Head=4、PCP=1。MRV2 关闭 batch-sharded sampling；MTP 使用 greedy 草稿采样，use_local_argmax_reduction=false、enable_adaptive_verification=false。主模型与 MTP 共享 Head 分片。',
      root:'#lmhead-split-steps'
    }
  };
  const $ = selector => document.querySelector(selector);
  const state = {module:'embedding', steps:{embedding:0,oproj:0,mlp:0,lmhead:0}};
  const text = (selector,value) => { $(selector).textContent=value; };
  function goStep(target, updateUrl=true) {
    const key=state.module, lesson=lessons[key], previous=state.steps[key];
    target=Math.max(0,Math.min(lesson.steps.length-1,target));
    const root=$(lesson.root);
    if(key==='lmhead') root.querySelector(`[data-step="${target}"]`).click();
    else for(let i=previous;i!==target;i+=Math.sign(target-previous)) root.querySelector(target>previous?lesson.next:lesson.prev).click();
    state.steps[key]=target;
    const [title,explanation,communication]=lesson.explanations[target];
    text('#step-count',`${String(target+1).padStart(2,'0')} / ${String(lesson.steps.length).padStart(2,'0')}`);
    text('#step-title',title);text('#step-explanation',explanation);text('#communication',communication);
    $('#step-list').innerHTML=lesson.steps.map((label,i)=>`<button type="button" data-jump="${i}" class="${i===target?'current':i<target?'done':''}" ${i===target?'aria-current="step"':''}><span>${String(i+1).padStart(2,'0')}</span>${label}</button>`).join('');
    $('#previous').disabled=target===0;$('#next').disabled=target===lesson.steps.length-1;
    $('#reset').disabled=target===0;
    if(updateUrl) history.replaceState(null,'',`#${key}/${target+1}`);
  }
  function selectModule(key, step, updateUrl=true) {
    if(!lessons[key]) key='embedding';
    state.module=key;const lesson=lessons[key];
    document.querySelectorAll('.demo-panel').forEach(panel=>panel.hidden=panel.id!==`panel-${key}`);
    document.querySelectorAll('[data-module]').forEach(button=>{const active=button.dataset.module===key;button.classList.toggle('active',active);button.setAttribute('aria-pressed',String(active));});
    text('#module-title',lesson.title);text('#module-role',lesson.role);text('#split-axis',lesson.axis);text('#takeaway',lesson.takeaway);text('#config-code',lesson.config);text('#config-note',lesson.note);
    document.title=`${{embedding:'Embedding',oproj:'OProj',mlp:'MLP',lmhead:'LM Head'}[key]} · 细粒度 TP 交互原理图`;
    goStep(step??state.steps[key],updateUrl);
  }
  function readHash(){const [key,step]=location.hash.slice(1).split('/');if(key==='lesson')return;selectModule(key,Number.isFinite(Number(step))&&step?Number(step)-1:0,false);}
  document.querySelectorAll('[data-module]').forEach(button=>button.addEventListener('click',()=>selectModule(button.dataset.module)));
  $('#step-list').addEventListener('click',event=>{const button=event.target.closest('[data-jump]');if(button)goStep(Number(button.dataset.jump));});
  $('#previous').addEventListener('click',()=>goStep(state.steps[state.module]-1));
  $('#next').addEventListener('click',()=>goStep(state.steps[state.module]+1));
  $('#reset').addEventListener('click',()=>goStep(0));
  document.addEventListener('keydown',event=>{if(event.altKey||event.ctrlKey||event.metaKey||event.shiftKey||['INPUT','TEXTAREA','SELECT'].includes(event.target.tagName))return;if(event.key==='ArrowRight'||event.key==='ArrowLeft'){event.preventDefault();goStep(state.steps[state.module]+(event.key==='ArrowRight'?1:-1));}});
  window.addEventListener('hashchange',readHash);
  let toastTimer;
  function toast(message){text('#toast',message);$('#toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('#toast').hidden=true,2400);}
  async function copy(value,success){try{if(navigator.clipboard&&window.isSecureContext)await navigator.clipboard.writeText(value);else{const field=document.createElement('textarea');field.value=value;field.style.cssText='position:fixed;left:-9999px';document.body.append(field);field.select();const done=document.execCommand('copy');field.remove();if(!done)throw Error('copy');}toast(success);}catch{toast('复制未完成，请手动选择地址栏或参数内容');}}
  $('#share').addEventListener('click',()=>copy(new URL(`#${state.module}/${state.steps[state.module]+1}`,location.href).href,'已复制当前模块与步骤的链接'));
  $('#copy-config').addEventListener('click',()=>copy(lessons[state.module].config,'已复制开启参数'));
  readHash();
})();
