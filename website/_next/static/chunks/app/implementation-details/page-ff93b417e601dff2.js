(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[810],{6885:(e,s,n)=>{Promise.resolve().then(n.bind(n,7139))},7139:(e,s,n)=>{"use strict";n.r(s),n.d(s,{default:()=>a});var i=n(2860),r=n(4836),t=n(2662),o=n(4612),l=n.n(o);function a(){return(0,i.jsxs)("main",{className:"flex min-h-screen flex-col items-center p-8 sm:p-24",children:[(0,i.jsx)("h1",{className:"text-4xl font-bold mb-6 text-center",children:"Implementation Details"}),(0,i.jsx)("div",{className:"w-full max-w-4xl mb-8",children:(0,i.jsxs)(t.Zp,{className:"p-6",children:[(0,i.jsx)("h2",{className:"text-2xl font-semibold mb-3",children:"Overview"}),(0,i.jsx)("p",{className:"mb-6",children:"This implementation of a Region Proposal Network (RPN) pipeline includes tensor generation, decoding, anchor encoding, and non-maximum suppression. The implementation follows a modular approach with each component designed to handle a specific part of the pipeline."}),(0,i.jsx)("h2",{className:"text-2xl font-semibold mb-3",children:"Key Constants"}),(0,i.jsx)("div",{className:"bg-gray-100 dark:bg-gray-800 p-4 rounded-md mb-6",children:(0,i.jsx)("pre",{className:"text-sm",children:"TARGET_SIZE = 200     # Target size for resized images\nGRID_NUMBER = 40      # Number of grid cells in each dimension\nPATCH_SIZE = 5        # Size of each patch\nTHRESHOLD = 0.5       # Confidence threshold for predictions"})}),(0,i.jsx)("h2",{className:"text-2xl font-semibold mb-3",children:"Complete Pipeline"}),(0,i.jsxs)("ol",{className:"list-decimal pl-5 mb-6",children:[(0,i.jsxs)("li",{className:"mb-2",children:[(0,i.jsx)("strong",{children:"Ground Truth Tensor Generation:"}),(0,i.jsxs)("ul",{className:"list-disc pl-5 mt-1",children:[(0,i.jsx)("li",{children:"Resize images to 200x200"}),(0,i.jsx)("li",{children:"Create a 40x40 grid of 5x5 patches"}),(0,i.jsx)("li",{children:"Generate existence and location tensors based on ground truth boxes"})]})]}),(0,i.jsxs)("li",{className:"mb-2",children:[(0,i.jsx)("strong",{children:"Tensor Decoding:"}),(0,i.jsxs)("ul",{className:"list-disc pl-5 mt-1",children:[(0,i.jsx)("li",{children:"Convert existence and location tensors back to bounding boxes"}),(0,i.jsx)("li",{children:"Apply confidence threshold to filter predictions"}),(0,i.jsx)("li",{children:"Scale coordinates back to original image dimensions"})]})]}),(0,i.jsxs)("li",{className:"mb-2",children:[(0,i.jsx)("strong",{children:"Anchor-Based Encoding:"}),(0,i.jsxs)("ul",{className:"list-disc pl-5 mt-1",children:[(0,i.jsx)("li",{children:"Use predefined anchor shapes at each patch center"}),(0,i.jsx)("li",{children:"Match anchors to ground truth boxes based on IoU"}),(0,i.jsx)("li",{children:"Encode ground truth boxes as offsets from anchors"})]})]}),(0,i.jsxs)("li",{className:"mb-2",children:[(0,i.jsx)("strong",{children:"Anchor-Based Decoding:"}),(0,i.jsxs)("ul",{className:"list-disc pl-5 mt-1",children:[(0,i.jsx)("li",{children:"Apply predicted offsets to anchor boxes"}),(0,i.jsx)("li",{children:"Convert decoded boxes back to original image scale"})]})]}),(0,i.jsxs)("li",{className:"mb-2",children:[(0,i.jsx)("strong",{children:"Non-Maximum Suppression:"}),(0,i.jsxs)("ul",{className:"list-disc pl-5 mt-1",children:[(0,i.jsx)("li",{children:"Sort boxes by confidence score"}),(0,i.jsx)("li",{children:"Iteratively keep highest-confidence boxes"}),(0,i.jsxs)("li",{children:["Remove overlapping boxes with IoU ",">"," threshold"]})]})]})]}),(0,i.jsx)("h2",{className:"text-2xl font-semibold mb-3",children:"Testing and Verification"}),(0,i.jsx)("p",{className:"mb-4",children:"All implemented functions pass the provided unit tests, confirming the correctness of the implementation. The main program has been verified to work correctly for:"}),(0,i.jsxs)("ul",{className:"list-disc pl-5 mb-6",children:[(0,i.jsx)("li",{children:"Job 1: Generating ground truth tensors"}),(0,i.jsx)("li",{children:"Job 2: Decoding ground truth tensors"}),(0,i.jsx)("li",{children:"Job 3: Performing anchor-based encoding"}),(0,i.jsx)("li",{children:"Job 4: Performing anchor-based decoding"}),(0,i.jsx)("li",{children:"Job 5: Applying non-maximum suppression"})]}),(0,i.jsx)("h2",{className:"text-2xl font-semibold mb-3",children:"Challenges and Solutions"}),(0,i.jsxs)("div",{className:"bg-gray-100 dark:bg-gray-800 p-4 rounded-md mb-6",children:[(0,i.jsx)("p",{className:"mb-2",children:(0,i.jsx)("strong",{children:"Challenge 1: Coordinate System Consistency"})}),(0,i.jsx)("p",{className:"mb-4",children:"Ensuring consistent coordinate system usage (bottom-left origin) throughout the pipeline. Solution: Careful documentation and consistent implementation of coordinate transformations."}),(0,i.jsx)("p",{className:"mb-2",children:(0,i.jsx)("strong",{children:"Challenge 2: Anchor Matching Strategy"})}),(0,i.jsx)("p",{className:"mb-4",children:"Determining the best strategy for matching anchors to ground truth boxes. Solution: Implemented IoU-based matching with best-match selection."}),(0,i.jsx)("p",{className:"mb-2",children:(0,i.jsx)("strong",{children:"Challenge 3: Efficient NMS Implementation"})}),(0,i.jsx)("p",{children:"Implementing NMS efficiently to handle large numbers of proposals. Solution: Sort-and-filter approach with early termination for efficiency."})]}),(0,i.jsx)("h2",{className:"text-2xl font-semibold mb-3",children:"Conclusion"}),(0,i.jsx)("p",{className:"mb-6",children:"The implementation successfully completes the core components of a Region Proposal Network pipeline. The code is modular, well-documented, and passes all provided tests, demonstrating a solid understanding of the RPN architecture and its components."}),(0,i.jsxs)("div",{className:"flex justify-between",children:[(0,i.jsx)(l(),{href:"/non-maximum-suppression",children:(0,i.jsx)(r.$,{variant:"outline",children:"Previous: Non-Maximum Suppression"})}),(0,i.jsx)(l(),{href:"/",children:(0,i.jsx)(r.$,{children:"Back to Home"})})]})]})})]})}},4836:(e,s,n)=>{"use strict";n.d(s,{$:()=>c});var i=n(2860),r=n(3200),t=n(4933),o=n(1073),l=n(2979);let a=(0,o.F)("inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",{variants:{variant:{default:"bg-primary text-primary-foreground shadow hover:bg-primary/90",destructive:"bg-destructive text-destructive-foreground shadow-sm hover:bg-destructive/90",outline:"border border-input bg-background shadow-sm hover:bg-accent hover:text-accent-foreground",secondary:"bg-secondary text-secondary-foreground shadow-sm hover:bg-secondary/80",ghost:"hover:bg-accent hover:text-accent-foreground",link:"text-primary underline-offset-4 hover:underline"},size:{default:"h-9 px-4 py-2",sm:"h-8 rounded-md px-3 text-xs",lg:"h-10 rounded-md px-8",icon:"h-9 w-9"}},defaultVariants:{variant:"default",size:"default"}}),c=r.forwardRef((e,s)=>{let{className:n,variant:r,size:o,asChild:c=!1,...d}=e,m=c?t.DX:"button";return(0,i.jsx)(m,{className:(0,l.cn)(a({variant:r,size:o,className:n})),ref:s,...d})});c.displayName="Button"},2662:(e,s,n)=>{"use strict";n.d(s,{Zp:()=>o});var i=n(2860),r=n(3200),t=n(2979);let o=r.forwardRef((e,s)=>{let{className:n,...r}=e;return(0,i.jsx)("div",{ref:s,className:(0,t.cn)("rounded-xl border bg-card text-card-foreground shadow",n),...r})});o.displayName="Card",r.forwardRef((e,s)=>{let{className:n,...r}=e;return(0,i.jsx)("div",{ref:s,className:(0,t.cn)("flex flex-col space-y-1.5 p-6",n),...r})}).displayName="CardHeader",r.forwardRef((e,s)=>{let{className:n,...r}=e;return(0,i.jsx)("div",{ref:s,className:(0,t.cn)("font-semibold leading-none tracking-tight",n),...r})}).displayName="CardTitle",r.forwardRef((e,s)=>{let{className:n,...r}=e;return(0,i.jsx)("div",{ref:s,className:(0,t.cn)("text-sm text-muted-foreground",n),...r})}).displayName="CardDescription",r.forwardRef((e,s)=>{let{className:n,...r}=e;return(0,i.jsx)("div",{ref:s,className:(0,t.cn)("p-6 pt-0",n),...r})}).displayName="CardContent",r.forwardRef((e,s)=>{let{className:n,...r}=e;return(0,i.jsx)("div",{ref:s,className:(0,t.cn)("flex items-center p-6 pt-0",n),...r})}).displayName="CardFooter"},2979:(e,s,n)=>{"use strict";n.d(s,{cn:()=>t});var i=n(9769),r=n(2460);function t(){for(var e=arguments.length,s=Array(e),n=0;n<e;n++)s[n]=arguments[n];return(0,r.QP)((0,i.$)(s))}}},e=>{var s=s=>e(e.s=s);e.O(0,[395,685,411,358],()=>s(6885)),_N_E=e.O()}]);