import React, { useState, useRef, useEffect } from 'react';
import LiveCursors from './LiveCursors';

interface CodeEditorProps {
  content: string;
  onChange: (content: string) => void;
  language: string;
  fileName: string;
  collaborators?: Array<{
    userId: string;
    userName: string;
    line: number;
    column: number;
    color: string;
  }>;
  onCursorChange?: (line: number, column: number) => void;
}

const CodeEditor: React.FC<CodeEditorProps> = ({ 
  content, 
  onChange, 
  language, 
  fileName, 
  collaborators = [],
  onCursorChange 
}) => {
  const [lineNumbers, setLineNumbers] = useState<number[]>([1]);
  const [cursorPosition, setCursorPosition] = useState({ line: 1, column: 1 });
  const [highlightedContent, setHighlightedContent] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const highlightRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const lines = content.split('\n').length;
    setLineNumbers(Array.from({ length: Math.max(lines, 20) }, (_, i) => i + 1));
    setHighlightedContent(syntaxHighlight(content));
  }, [content, language]);

  const handleScroll = (e: React.UIEvent<HTMLTextAreaElement>) => {
    const lineNumbersEl = document.querySelector('.line-numbers');
    const highlightEl = highlightRef.current;
    
    if (lineNumbersEl) {
      lineNumbersEl.scrollTop = e.currentTarget.scrollTop;
    }
    if (highlightEl) {
      highlightEl.scrollTop = e.currentTarget.scrollTop;
      highlightEl.scrollLeft = e.currentTarget.scrollLeft;
    }
  };

  const handleCursorMove = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const textarea = e.target;
    const cursorPos = textarea.selectionStart;
    const textBeforeCursor = content.substring(0, cursorPos);
    const lines = textBeforeCursor.split('\n');
    const line = lines.length;
    const column = lines[lines.length - 1].length + 1;
    
    setCursorPosition({ line, column });
    onCursorChange?.(line, column);
  };

  const handleContentChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    onChange(e.target.value);
    handleCursorMove(e);
  };

  const syntaxHighlight = (code: string) => {
    let highlighted = code;
    
    if (language === 'rust') {
      highlighted = highlighted
        .replace(/(fn|let|mut|pub|struct|impl|enum|match|if|else|while|for|loop|break|continue|return|use|mod|crate|super|self|const|static|unsafe|async|await|move|ref|dyn|where|type|trait|macro_rules!)\b/g, '<span class="text-purple-400 font-medium">$1</span>')
        .replace(/(String|Vec|Option|Result|Box|Rc|Arc|HashMap|BTreeMap|i32|i64|u32|u64|f32|f64|bool|char|str|usize|isize)\b/g, '<span class="text-blue-400">$1</span>')
        .replace(/(println!|print!|dbg!|panic!|todo!|unimplemented!|unreachable!)\b/g, '<span class="text-green-400">$1</span>')
        .replace(/(".*?")/g, '<span class="text-yellow-400">$1</span>')
        .replace(/(\/\/.*$)/gm, '<span class="text-gray-500 italic">$1</span>')
        .replace(/(\/\*[\s\S]*?\*\/)/g, '<span class="text-gray-500 italic">$1</span>');
    }
    
    if (language === 'dart') {
      highlighted = highlighted
        .replace(/(class|extends|implements|with|abstract|static|final|const|var|dynamic|void|int|double|String|bool|List|Map|Set|Future|Stream|async|await|yield|import|export|library|part|show|hide|as|if|else|for|while|do|switch|case|default|break|continue|return|try|catch|finally|throw|rethrow|assert)\b/g, '<span class="text-purple-400 font-medium">$1</span>')
        .replace(/(Widget|StatelessWidget|StatefulWidget|State|BuildContext|MaterialApp|Scaffold|AppBar|Container|Column|Row|Text|Button|FloatingActionButton)\b/g, '<span class="text-blue-400">$1</span>')
        .replace(/(@override|@required|@deprecated)\b/g, '<span class="text-orange-400">$1</span>')
        .replace(/(".*?"|'.*?')/g, '<span class="text-yellow-400">$1</span>')
        .replace(/(\/\/.*$)/gm, '<span class="text-gray-500 italic">$1</span>')
        .replace(/(\/\*[\s\S]*?\*\/)/g, '<span class="text-gray-500 italic">$1</span>');
    }
    
    if (language === 'kotlin') {
      highlighted = highlighted
        .replace(/(class|interface|object|enum|data|sealed|abstract|open|final|override|private|protected|public|internal|fun|val|var|const|lateinit|lazy|by|delegate|if|else|when|for|while|do|try|catch|finally|throw|return|break|continue|import|package)\b/g, '<span class="text-purple-400 font-medium">$1</span>')
        .replace(/(String|Int|Long|Float|Double|Boolean|Char|Byte|Short|Any|Unit|Nothing|List|MutableList|Set|MutableSet|Map|MutableMap|Array|IntArray)\b/g, '<span class="text-blue-400">$1</span>')
        .replace(/(@\w+)\b/g, '<span class="text-orange-400">$1</span>')
        .replace(/(".*?")/g, '<span class="text-yellow-400">$1</span>')
        .replace(/(\/\/.*$)/gm, '<span class="text-gray-500 italic">$1</span>')
        .replace(/(\/\*[\s\S]*?\*\/)/g, '<span class="text-gray-500 italic">$1</span>');
    }
    
    if (language === 'typescript' || language === 'javascript') {
      highlighted = highlighted
        .replace(/(import|export|from|default|const|let|var|function|class|interface|type|enum|namespace|async|await|return|if|else|for|while|do|switch|case|break|continue|try|catch|finally|throw|new|this|super|extends|implements|public|private|protected|static|readonly|abstract)\b/g, '<span class="text-purple-400 font-medium">$1</span>')
        .replace(/(string|number|boolean|object|undefined|null|void|any|unknown|never|Array|Promise|React|Component|useState|useEffect|useCallback|useMemo|StyleSheet|View|Text|TouchableOpacity|SafeAreaView|ScrollView)\b/g, '<span class="text-blue-400">$1</span>')
        .replace(/(@\w+)\b/g, '<span class="text-orange-400">$1</span>')
        .replace(/(".*?"|'.*?'|`.*?`)/g, '<span class="text-yellow-400">$1</span>')
        .replace(/(\/\/.*$)/gm, '<span class="text-gray-500 italic">$1</span>')
        .replace(/(\/\*[\s\S]*?\*\/)/g, '<span class="text-gray-500 italic">$1</span>');
    }
    
    if (language === 'java') {
      highlighted = highlighted
        .replace(/(public|private|protected|static|final|abstract|class|interface|extends|implements|import|package|new|this|super|return|if|else|for|while|do|switch|case|break|continue|try|catch|finally|throw|synchronized|volatile|transient|native|strictfp)\b/g, '<span class="text-purple-400 font-medium">$1</span>')
        .replace(/(String|int|long|float|double|boolean|char|byte|short|void|Object|List|Map|Set|Array|ArrayList|HashMap|HashSet|Intent|Bundle|Activity|Application|Context|View|TextView|Button|LinearLayout|RelativeLayout)\b/g, '<span class="text-blue-400">$1</span>')
        .replace(/(@\w+)\b/g, '<span class="text-orange-400">$1</span>')
        .replace(/(".*?")/g, '<span class="text-yellow-400">$1</span>')
        .replace(/(\/\/.*$)/gm, '<span class="text-gray-500 italic">$1</span>')
        .replace(/(\/\*[\s\S]*?\*\/)/g, '<span class="text-gray-500 italic">$1</span>');
    }
    
    if (language === 'objective-c') {
      highlighted = highlighted
        .replace(/(@interface|@implementation|@end|@property|@synthesize|@dynamic|@protocol|@optional|@required|@class|@import|#import|#include|if|else|for|while|do|switch|case|break|continue|return|static|const|extern|typedef|struct|enum|union)\b/g, '<span class="text-purple-400 font-medium">$1</span>')
        .replace(/(NSString|NSArray|NSDictionary|NSNumber|NSObject|UIView|UIViewController|UIApplication|UIWindow|BOOL|NSInteger|CGFloat|CGRect|CGPoint|CGSize|id|void|int|float|double|char|long|short)\b/g, '<span class="text-blue-400">$1</span>')
        .replace(/(@".*?")/g, '<span class="text-yellow-400">$1</span>')
        .replace(/(\/\/.*$)/gm, '<span class="text-gray-500 italic">$1</span>')
        .replace(/(\/\*[\s\S]*?\*\/)/g, '<span class="text-gray-500 italic">$1</span>');
    }
    
    if (language === 'toml') {
      highlighted = highlighted
        .replace(/(\[.*?\])/g, '<span class="text-blue-400 font-medium">$1</span>')
        .replace(/^(\w+)\s*=/gm, '<span class="text-purple-400">$1</span>=')
        .replace(/(".*?")/g, '<span class="text-yellow-400">$1</span>')
        .replace(/(#.*$)/gm, '<span class="text-gray-500 italic">$1</span>');
    }
    
    if (language === 'json') {
      highlighted = highlighted
        .replace(/(".*?")(\s*:)/g, '<span class="text-blue-400">$1</span>$2')
        .replace(/:\s*(".*?")/g, ': <span class="text-yellow-400">$1</span>')
        .replace(/:\s*(true|false|null)\b/g, ': <span class="text-orange-400">$1</span>')
        .replace(/:\s*(\d+\.?\d*)/g, ': <span class="text-green-400">$1</span>');
    }
    
    if (language === 'xml') {
      highlighted = highlighted
        .replace(/(&lt;\/?)([\w-]+)/g, '$1<span class="text-blue-400">$2</span>')
        .replace(/([\w-]+)(=)/g, '<span class="text-purple-400">$1</span>$2')
        .replace(/=(".*?")/g, '=<span class="text-yellow-400">$1</span>')
        .replace(/(&lt;!--[\s\S]*?--&gt;)/g, '<span class="text-gray-500 italic">$1</span>');
    }
    
    return highlighted;
  };

  return (
    <div className="flex h-full bg-gray-900 text-gray-100 font-mono text-sm relative">
      <div className="line-numbers bg-gray-800 px-4 py-4 text-gray-500 text-right select-none overflow-hidden border-r border-gray-700 min-w-[4rem]">
        {lineNumbers.map((num) => (
          <div key={num} className="leading-6">
            {num}
          </div>
        ))}
      </div>
      <div className="flex-1 relative overflow-hidden">
        <LiveCursors cursors={collaborators} currentFile={fileName} />
        
        {/* Syntax highlighted background */}
        <div 
          ref={highlightRef}
          className="absolute inset-0 p-4 pointer-events-none whitespace-pre-wrap break-words font-mono text-sm leading-6 overflow-auto"
          dangerouslySetInnerHTML={{ __html: highlightedContent }}
          style={{ 
            color: 'transparent',
            background: 'transparent',
            zIndex: 1
          }}
        />
        
        {/* Transparent textarea overlay */}
        <textarea
          ref={textareaRef}
          className="absolute inset-0 w-full h-full p-4 bg-transparent resize-none outline-none font-mono text-sm leading-6 text-transparent caret-white selection:bg-blue-500/30"
          value={content}
          onChange={handleContentChange}
          onSelect={handleCursorMove}
          onScroll={handleScroll}
          spellCheck={false}
          placeholder={`Start coding in ${fileName}...`}
          style={{
            tabSize: 4,
            zIndex: 2
          }}
        />
      </div>
    </div>
  );
};

export default CodeEditor;