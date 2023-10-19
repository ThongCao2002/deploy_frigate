import{h as M,u as T,I as R,o as e,H as m,D as A,A as x,d as h,k as X,c as G}from"./index-1b1f5707.js";import{L as B}from"./Link-0ea49822.js";import{T as _,a as k,b as p,c as a,d as C,e as n}from"./Table-024ade0a.js";import{T as Q}from"./TimeAgo-257a48f4.js";import{c as U}from"./index-2f918ad2.js";import{a as b}from"./About-5c887223.js";const y=Object.freeze({});function de(){const[l,f]=M({showFfprobe:!1,ffprobe:""}),{data:c}=T("config"),{value:{payload:I}}=R("stats"),{data:L}=T("stats"),{cpu_usages:t,gpu_usages:v,bandwidth_usages:P,detectors:g,service:j={},detection_fps:Y,processes:u,...r}=I||L||y,D=Object.keys(g||y),H=Object.keys(v||y),q=Object.keys(r||y),K=Object.keys(u||y),{data:F}=T("go2rtc"),$=async(i,s)=>{s&&s.stopPropagation(),f({...l,showFfprobe:!0});const d=await G.get("ffprobe",{params:{paths:`camera:${i}`}});d.status===200?f({...l,showFfprobe:!0,ffprobe:d.data}):f({...l,showFfprobe:!0,ffprobe:"There was an error getting the ffprobe output."})},J=async()=>{U(JSON.stringify(l.ffprobe).replace(/[\\\s]+/gi,"")),f({...l,ffprobe:"",showFfprobe:!1})},W=async i=>{i&&i.stopPropagation();const s=await G.get("vainfo");s.status===200?f({...l,showVainfo:!0,vainfo:s.data}):f({...l,showVainfo:!0,vainfo:"There was an error getting the vainfo output."})},E=async()=>{U(JSON.stringify(l.vainfo).replace(/[\\\s]+/gi,"")),f({...l,vainfo:"",showVainfo:!1})};return e("div",{className:"space-y-4 p-2 px-4",children:[e("div",{className:"flex justify-between",children:[e(m,{children:["Hệ thống ",e("span",{className:"text-sm",children:j.version})]}),c&&e("span",{class:"p-1",children:["go2rtc ",F&&`${F.version} `,e(B,{className:"text-blue-500 hover:underline",target:"_blank",rel:"noopener noreferrer",href:"/live/webrtc/",children:"Bảng điều khiển"})]})]}),j.last_updated&&e("p",{children:e("span",{children:["Cập nhật lần cuối vào: ",e(Q,{time:j.last_updated*1e3,dense:!0})]})}),l.showFfprobe&&e(A,{children:[e("div",{className:"p-4 mb-2 max-h-96 whitespace-pre-line overflow-auto",children:[e(m,{size:"lg",children:"Ffprobe Output"}),l.ffprobe!=""?e("div",{children:l.ffprobe.map((i,s)=>e("div",{className:"mb-2 max-h-96 whitespace-pre-line",children:[e("div",{children:["Stream ",s,":"]}),e("div",{className:"px-2",children:["Trả về Code: ",i.return_code]}),e("br",{}),i.return_code==0?e("div",{children:i.stdout.streams.map((d,o)=>e("div",{className:"px-2",children:d.width?e("div",{children:[e("div",{children:"Video:"}),e("br",{}),e("div",{children:["Codec: ",d.codec_long_name]}),e("div",{children:["Phân giải: ",d.width,"x",d.height]}),e("div",{children:["FPS: ",d.avg_frame_rate=="0/0"?"Unknown":d.avg_frame_rate]}),e("br",{})]}):e("div",{children:[e("div",{children:"Âm thanh:"}),e("br",{}),e("div",{children:["Codec: ",d.codec_long_name]}),e("br",{})]})},o))}):e("div",{className:"px-2",children:e("div",{children:["Lỗi: ",i.stderr]})})]},s))}):e(x,{})]}),e("div",{className:"p-2 flex justify-start flex-row-reverse space-x-2",children:[e(h,{className:"ml-2",onClick:()=>J(),type:"text",children:"Sao chép"}),e(h,{className:"ml-2",onClick:()=>f({...l,ffprobe:"",showFfprobe:!1}),type:"text",children:"Đóng"})]})]}),l.showVainfo&&e(A,{children:[e("div",{className:"p-4 overflow-auto whitespace-pre-line",children:[e(m,{size:"lg",children:"Vainfo Output"}),l.vainfo!=""?e("div",{className:"mb-2 max-h-96 whitespace-pre-line",children:[e("div",{className:"",children:["Trả về Code: ",l.vainfo.return_code]}),e("br",{}),e("div",{className:"",children:["Tiến trình ",l.vainfo.return_code==0?"Output":"Error",":"]}),e("br",{}),e("div",{children:l.vainfo.return_code==0?l.vainfo.stdout:l.vainfo.stderr})]}):e(x,{})]}),e("div",{className:"p-2 flex justify-start flex-row-reverse space-x-2 whitespace-pre-wrap",children:[e(h,{className:"ml-2",onClick:()=>E(),type:"text",children:"Sao chép"}),e(h,{className:"ml-2",onClick:()=>f({...l,vainfo:"",showVainfo:!1}),type:"text",children:"Đóng"})]})]}),g?e(X,{children:[e("div",{className:"flex justify-start",children:[e(m,{className:"self-center",size:"lg",children:"Bộ nhận diện"}),e(h,{className:"rounded-full",type:"text",color:"gray","aria-label":"Lượng tài nguyên hiện tại của mỗi tiến trình điều khiển bộ nhận diện. CPU % thể hiện cho một lõi.",children:e(b,{className:"w-5"})})]}),e("div",{"data-testid":"detectors",className:"grid grid-cols-1 3xl:grid-cols-3 md:grid-cols-2 gap-4",children:D.map(i=>{var s,d,o;return e("div",{className:"dark:bg-gray-800 shadow-md hover:shadow-lg rounded-lg transition-shadow",children:[e("div",{className:"text-lg flex justify-between p-4",children:i}),e("div",{className:"p-2",children:e(_,{className:"w-full",children:[e(k,{children:e(p,{children:[e(a,{children:"P-ID"}),e(a,{children:"Tốc độ nhận diện"}),e(a,{children:"CPU %"}),e(a,{children:"Bộ nhớ %"}),c.telemetry.network_bandwidth&&e(a,{children:"Băng thông mạng"})]})}),e(C,{children:e(p,{children:[e(n,{children:g[i].pid}),e(n,{children:[g[i].inference_speed," ms"]}),e(n,{children:[((s=t[g[i].pid])==null?void 0:s.cpu)||"- ","%"]}),e(n,{children:[((d=t[g[i].pid])==null?void 0:d.mem)||"- ","%"]}),c.telemetry.network_bandwidth&&e(n,{children:[((o=P[g[i].pid])==null?void 0:o.bandwidth)||"- ","KB/s"]})]})})]})})]},i)})}),e("div",{className:"text-lg flex justify-between",children:[e("div",{className:"flex justify-start",children:[e(m,{className:"self-center",size:"lg",children:"GPUs"}),e(h,{className:"rounded-full",type:"text",color:"gray","aria-label":"Lượng tài nguyên hiện tại của mỗi GPU. GPU Intel không hỗ trợ thống kê bộ nhớ.",children:e(b,{className:"w-5"})})]}),e(h,{onClick:i=>W(i),children:"vainfo"})]}),v?e("div",{"data-testid":"gpus",className:"grid grid-cols-1 3xl:grid-cols-3 md:grid-cols-2 gap-4",children:H.map(i=>e("div",{className:"dark:bg-gray-800 shadow-md hover:shadow-lg rounded-lg transition-shadow",children:[e("div",{className:"text-lg flex justify-between p-4",children:i}),e("div",{className:"p-2",children:v[i].gpu==-1?e("div",{className:"p-4",children:"Xảy ra lỗi khi truy xuất thống kê sử dụng. Điều này không có nghĩa là tăng cường phần cứng không hoạt động. Có thể GPU không hỗ trợ hoặc hệ thống không có quyền truy cập để cập nhật thống kê."}):e(_,{className:"w-full",children:[e(k,{children:e(p,{children:[e(a,{children:"GPU %"}),e(a,{children:"Bộ nhớ %"})]})}),e(C,{children:e(p,{children:[e(n,{children:v[i].gpu}),e(n,{children:v[i].mem})]})})]})})]},i))}):e("div",{className:"p-4",children:e(B,{href:"https://docs.frigate.video/configuration/hardware_acceleration",children:"Tăng tốc phần cứng chưa được thiết lập, hãy xem tài liệu để thiết lập tăng tốc phần cứng."})}),e("div",{className:"flex justify-start",children:[e(m,{className:"self-center",size:"lg",children:"Cameras"}),e(h,{className:"rounded-full",type:"text",color:"gray","aria-label":"Lượng tài nguyên hiện tại của mỗi tiến trình tương tác với Stream Camera. CPU % đại diện cho một lõi.",children:e(b,{className:"w-5"})})]}),r?e("div",{"data-testid":"cameras",className:"grid grid-cols-1 3xl:grid-cols-3 md:grid-cols-2 gap-4",children:q.map(i=>{var s,d,o,w,S,O,V,z;return c.cameras[i].enabled&&e("div",{className:"dark:bg-gray-800 shadow-md hover:shadow-lg rounded-lg transition-shadow",children:[e("div",{className:"capitalize text-lg flex justify-between p-4",children:[e(B,{href:`/cameras/${i}`,children:i.replaceAll("_"," ")}),e(h,{onClick:N=>$(i,N),children:"ffprobe"})]}),e("div",{className:"p-2",children:e(_,{className:"w-full",children:[e(k,{children:e(p,{children:[e(a,{children:"Tiến trình"}),e(a,{children:"P-ID"}),e(a,{children:"FPS"}),e(a,{children:"CPU %"}),e(a,{children:"Bộ nhớ %"}),c.telemetry.network_bandwidth&&e(a,{children:"Băng thông mạng"})]})}),e(C,{children:[e(p,{index:"0",children:[e(n,{children:["ffmpeg",e(h,{className:"rounded-full",type:"text",color:"gray","aria-label":(s=t[r[i].ffmpeg_pid])==null?void 0:s.cmdline,onClick:()=>{var N;return U((N=t[r[i].ffmpeg_pid])==null?void 0:N.cmdline)},children:e(b,{className:"w-3"})})]}),e(n,{children:r[i].ffmpeg_pid||"- "}),e(n,{children:r[i].camera_fps||"- "}),e(n,{children:[((d=t[r[i].ffmpeg_pid])==null?void 0:d.cpu)||"- ","%"]}),e(n,{children:[((o=t[r[i].ffmpeg_pid])==null?void 0:o.mem)||"- ","%"]}),c.telemetry.network_bandwidth&&e(n,{children:[((w=P[r[i].ffmpeg_pid])==null?void 0:w.bandwidth)||"- ","KB/s"]})]},"ffmpeg"),e(p,{index:"1",children:[e(n,{children:"Ghi nhận"}),e(n,{children:r[i].capture_pid||"- "}),e(n,{children:r[i].process_fps||"- "}),e(n,{children:[((S=t[r[i].capture_pid])==null?void 0:S.cpu)||"- ","%"]}),e(n,{children:[((O=t[r[i].capture_pid])==null?void 0:O.mem)||"- ","%"]}),c.telemetry.network_bandwidth&&e(n,{children:"-"})]},"capture"),e(p,{index:"2",children:[e(n,{children:"Nhận diện"}),e(n,{children:r[i].pid||"- "}),(()=>r[i].pid&&r[i].detection_enabled==1?e(n,{children:[r[i].detection_fps," (",r[i].skipped_fps," bỏ qua)"]}):r[i].pid&&r[i].detection_enabled==0?e(n,{children:"Vô hiệu hóa"}):e(n,{children:"- "}))(),e(n,{children:[((V=t[r[i].pid])==null?void 0:V.cpu)||"- ","%"]}),e(n,{children:[((z=t[r[i].pid])==null?void 0:z.mem)||"- ","%"]}),c.telemetry.network_bandwidth&&e(n,{children:"-"})]},"detect")]})]})})]},i)})}):e(x,{}),e("div",{className:"flex justify-start",children:[e(m,{className:"self-center",size:"lg",children:"Tiến trình khác"}),e(h,{className:"rounded-full",type:"text",color:"gray","aria-label":"Momentary resource usage for other important processes. CPU % is for a single core.",children:e(b,{className:"w-5"})})]}),e("div",{"data-testid":"cameras",className:"grid grid-cols-1 3xl:grid-cols-3 md:grid-cols-2 gap-4",children:K.map(i=>{var s,d,o,w;return e("div",{className:"dark:bg-gray-800 shadow-md hover:shadow-lg rounded-lg transition-shadow",children:[e("div",{className:"capitalize text-lg flex justify-between p-4",children:e("div",{className:"text-lg flex justify-between",children:i})}),e("div",{className:"p-2",children:e(_,{className:"w-full",children:[e(k,{children:e(p,{children:[e(a,{children:"P-ID"}),e(a,{children:"CPU %"}),e(a,{children:"Avg CPU %"}),e(a,{children:"Bộ nhớ %"}),c.telemetry.network_bandwidth&&e(a,{children:"Băng thông mạng "})]})}),e(C,{children:e(p,{index:"0",children:[e(n,{children:u[i].pid||"- "}),e(n,{children:[((s=t[u[i].pid])==null?void 0:s.cpu)||"- ","%"]}),e(n,{children:[((d=t[u[i].pid])==null?void 0:d.cpu_average)||"- ","%"]}),e(n,{children:[((o=t[u[i].pid])==null?void 0:o.mem)||"- ","%"]}),c.telemetry.network_bandwidth&&e(n,{children:[((w=P[u[i].pid])==null?void 0:w.bandwidth)||"- ","KB/s"]})]},"other")})]})})]},i)})}),e("p",{children:["Thống kê hệ thống tự động cập nhật sau ",c.mqtt.stats_interval," giây."]})]}):e("div",{children:e(x,{})})]})}export{de as default};