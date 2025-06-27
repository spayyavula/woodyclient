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
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const highlightRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const lines = content.split('\n').length;
    setLineNumbers(Array.from({ length: Math.max(lines, 20) }, (_, i) => i + 1));
  }, [content]);

  const handleScroll = (e: React.UIEvent<HTMLTextAreaElement>) => {
    const lineNumbersEl = document.querySelector('.line-numbers');
    if (lineNumbersEl) {
      lineNumbersEl.scrollTop = e.currentTarget.scrollTop;
    }
    if (highlightRef.current) {
      highlightRef.current.scrollTop = e.currentTarget.scrollTop;
      highlightRef.current.scrollLeft = e.currentTarget.scrollLeft;
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
    if (language === 'rust') {
      return code
        // Comments first (VS Code: #6A9955)
        .replace(/(\/\/.*$)/gm, '<span style="color: #6A9955; font-style: italic;">$1</span>')
        .replace(/(\/\*[\s\S]*?\*\/)/g, '<span style="color: #6A9955; font-style: italic;">$1</span>')
        // Keywords (VS Code: #569CD6)
        .replace(/\b(fn|let|mut|pub|struct|impl|enum|match|if|else|while|for|loop|break|continue|return|use|mod|crate|super|self|const|static|unsafe|async|await|move|ref|dyn|where|type|trait|macro_rules!)\b/g, '<span style="color: #569CD6; font-weight: 500;">$1</span>')
        // Types (VS Code: #4EC9B0)
        .replace(/\b(String|Vec|Option|Result|Box|Rc|Arc|HashMap|BTreeMap|i32|i64|u32|u64|f32|f64|bool|char|str|usize|isize)\b/g, '<span style="color: #4EC9B0;">$1</span>')
        // Macros (VS Code: #DCDCAA)
        .replace(/\b(println!|print!|dbg!|panic!|todo!|unimplemented!|unreachable!)\b/g, '<span style="color: #DCDCAA;">$1</span>')
        // Strings (VS Code: #CE9178)
        .replace(/("(?:[^"\\]|\\.)*")/g, '<span style="color: #CE9178;">$1</span>')
        // Numbers (VS Code: #B5CEA8)
        .replace(/\b(\d+\.?\d*)\b/g, '<span style="color: #B5CEA8;">$1</span>')
        // Function names (VS Code: #DCDCAA)
        .replace(/\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(?=\()/g, '<span style="color: #DCDCAA;">$1</span>');
    }
    
    if (language === 'dart') {
      return code
        // Comments (VS Code: #6A9955)
        .replace(/(\/\/.*$)/gm, '<span style="color: #6A9955; font-style: italic;">$1</span>')
        .replace(/(\/\*[\s\S]*?\*\/)/g, '<span style="color: #6A9955; font-style: italic;">$1</span>')
        // Keywords (VS Code: #569CD6)
        .replace(/\b(class|extends|implements|with|abstract|static|final|const|var|dynamic|void|int|double|String|bool|List|Map|Set|Future|Stream|async|await|yield|import|export|library|part|show|hide|as|if|else|for|while|do|switch|case|default|break|continue|return|try|catch|finally|throw|rethrow|assert)\b/g, '<span style="color: #569CD6; font-weight: 500;">$1</span>')
        // Flutter/Dart types (VS Code: #4EC9B0)
        .replace(/\b(Widget|StatelessWidget|StatefulWidget|State|BuildContext|MaterialApp|Scaffold|AppBar|Container|Column|Row|Text|Button|FloatingActionButton)\b/g, '<span style="color: #4EC9B0;">$1</span>')
        // Annotations (VS Code: #9CDCFE)
        .replace(/(@override|@required|@deprecated)/g, '<span style="color: #9CDCFE;">$1</span>')
        // Strings (VS Code: #CE9178)
        .replace(/("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/g, '<span style="color: #CE9178;">$1</span>')
        // Numbers (VS Code: #B5CEA8)
        .replace(/\b(\d+\.?\d*)\b/g, '<span style="color: #B5CEA8;">$1</span>')
        // Function names (VS Code: #DCDCAA)
        .replace(/\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(?=\()/g, '<span style="color: #DCDCAA;">$1</span>');
    }
    
    if (language === 'kotlin') {
      return code
        // Comments (VS Code: #6A9955)
        .replace(/(\/\/.*$)/gm, '<span style="color: #6A9955; font-style: italic;">$1</span>')
        .replace(/(\/\*[\s\S]*?\*\/)/g, '<span style="color: #6A9955; font-style: italic;">$1</span>')
        // Keywords (VS Code: #569CD6)
        .replace(/\b(class|interface|object|enum|data|sealed|abstract|open|final|override|private|protected|public|internal|fun|val|var|const|lateinit|lazy|by|delegate|if|else|when|for|while|do|try|catch|finally|throw|return|break|continue|import|package)\b/g, '<span style="color: #569CD6; font-weight: 500;">$1</span>')
        // Types (VS Code: #4EC9B0)
        .replace(/\b(String|Int|Long|Float|Double|Boolean|Char|Byte|Short|Any|Unit|Nothing|List|MutableList|Set|MutableSet|Map|MutableMap|Array|IntArray)\b/g, '<span style="color: #4EC9B0;">$1</span>')
        // Annotations (VS Code: #9CDCFE)
        .replace(/(@\w+)/g, '<span style="color: #9CDCFE;">$1</span>')
        // Strings (VS Code: #CE9178)
        .replace(/("(?:[^"\\]|\\.)*")/g, '<span style="color: #CE9178;">$1</span>')
        // Numbers (VS Code: #B5CEA8)
        .replace(/\b(\d+\.?\d*)\b/g, '<span style="color: #B5CEA8;">$1</span>')
        // Function names (VS Code: #DCDCAA)
        .replace(/\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(?=\()/g, '<span style="color: #DCDCAA;">$1</span>');
    }
    
    if (language === 'typescript' || language === 'javascript') {
      return code
        // Comments (VS Code: #6A9955)
        .replace(/(\/\/.*$)/gm, '<span style="color: #6A9955; font-style: italic;">$1</span>')
        .replace(/(\/\*[\s\S]*?\*\/)/g, '<span style="color: #6A9955; font-style: italic;">$1</span>')
        // Keywords (VS Code: #569CD6)
        .replace(/\b(import|export|from|default|const|let|var|function|class|interface|type|enum|namespace|async|await|return|if|else|for|while|do|switch|case|break|continue|try|catch|finally|throw|new|this|super|extends|implements|public|private|protected|static|readonly|abstract)\b/g, '<span style="color: #569CD6; font-weight: 500;">$1</span>')
        // Types and React (VS Code: #4EC9B0)
        .replace(/\b(string|number|boolean|object|undefined|null|void|any|unknown|never|Array|Promise|React|Component|useState|useEffect|useCallback|useMemo|StyleSheet|View|Text|TouchableOpacity|SafeAreaView|ScrollView)\b/g, '<span style="color: #4EC9B0;">$1</span>')
        // Decorators (VS Code: #9CDCFE)
        .replace(/(@\w+)/g, '<span style="color: #9CDCFE;">$1</span>')
        // Strings (VS Code: #CE9178)
        .replace(/("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|`(?:[^`\\]|\\.)*`)/g, '<span style="color: #CE9178;">$1</span>')
        // Numbers (VS Code: #B5CEA8)
        .replace(/\b(\d+\.?\d*)\b/g, '<span style="color: #B5CEA8;">$1</span>')
        // Function names (VS Code: #DCDCAA)
        .replace(/\b([a-zA-Z_$][a-zA-Z0-9_$]*)\s*(?=\()/g, '<span style="color: #DCDCAA;">$1</span>');
    }
    
    if (language === 'java') {
      return code
        // Comments (VS Code: #6A9955)
        .replace(/(\/\/.*$)/gm, '<span style="color: #6A9955; font-style: italic;">$1</span>')
        .replace(/(\/\*[\s\S]*?\*\/)/g, '<span style="color: #6A9955; font-style: italic;">$1</span>')
        // Keywords (VS Code: #569CD6)
        .replace(/\b(public|private|protected|static|final|abstract|class|interface|extends|implements|import|package|new|this|super|return|if|else|for|while|do|switch|case|break|continue|try|catch|finally|throw|synchronized|volatile|transient|native|strictfp)\b/g, '<span style="color: #569CD6; font-weight: 500;">$1</span>')
        // Types (VS Code: #4EC9B0)
        .replace(/\b(String|int|long|float|double|boolean|char|byte|short|void|Object|List|Map|Set|Array|ArrayList|HashMap|HashSet|Intent|Bundle|Activity|Application|Context|View|TextView|Button|LinearLayout|RelativeLayout)\b/g, '<span style="color: #4EC9B0;">$1</span>')
        // Annotations (VS Code: #9CDCFE)
        .replace(/(@\w+)/g, '<span style="color: #9CDCFE;">$1</span>')
        // Strings (VS Code: #CE9178)
        .replace(/("(?:[^"\\]|\\.)*")/g, '<span style="color: #CE9178;">$1</span>')
        // Numbers (VS Code: #B5CEA8)
        .replace(/\b(\d+\.?\d*)\b/g, '<span style="color: #B5CEA8;">$1</span>')
        // Function names (VS Code: #DCDCAA)
        .replace(/\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(?=\()/g, '<span style="color: #DCDCAA;">$1</span>');
    }
    
    if (language === 'objective-c') {
      return code
        // Comments (VS Code: #6A9955)
        .replace(/(\/\/.*$)/gm, '<span style="color: #6A9955; font-style: italic;">$1</span>')
        .replace(/(\/\*[\s\S]*?\*\/)/g, '<span style="color: #6A9955; font-style: italic;">$1</span>')
        // Keywords (VS Code: #569CD6)
        .replace(/\b(@interface|@implementation|@end|@property|@synthesize|@dynamic|@protocol|@optional|@required|@class|@import|#import|#include|if|else|for|while|do|switch|case|break|continue|return|static|const|extern|typedef|struct|enum|union)\b/g, '<span style="color: #569CD6; font-weight: 500;">$1</span>')
        // Types (VS Code: #4EC9B0)
        .replace(/\b(NSString|NSArray|NSDictionary|NSNumber|NSObject|UIView|UIViewController|UIApplication|UIWindow|BOOL|NSInteger|CGFloat|CGRect|CGPoint|CGSize|id|void|int|float|double|char|long|short)\b/g, '<span style="color: #4EC9B0;">$1</span>')
        // Strings (VS Code: #CE9178)
        .replace(/(@"(?:[^"\\]|\\.)*")/g, '<span style="color: #CE9178;">$1</span>')
        // Numbers (VS Code: #B5CEA8)
        .replace(/\b(\d+\.?\d*)\b/g, '<span style="color: #B5CEA8;">$1</span>')
        // Function names (VS Code: #DCDCAA)
        .replace(/\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(?=\()/g, '<span style="color: #DCDCAA;">$1</span>');
    }
    
    if (language === 'toml') {
      return code
        // Comments (VS Code: #6A9955)
        .replace(/(#.*$)/gm, '<span style="color: #6A9955; font-style: italic;">$1</span>')
        // Section headers (VS Code: #569CD6)
        .replace(/(\[.*\])/g, '<span style="color: #569CD6; font-weight: 500;">$1</span>')
        // Keys (VS Code: #9CDCFE)
        .replace(/^(\w+)\s*=/gm, '<span style="color: #9CDCFE;">$1</span>=')
        // Strings (VS Code: #CE9178)
        .replace(/("(?:[^"\\]|\\.)*")/g, '<span style="color: #CE9178;">$1</span>')
        // Numbers (VS Code: #B5CEA8)
        .replace(/\b(\d+\.?\d*)\b/g, '<span style="color: #B5CEA8;">$1</span>');
    }
    
    if (language === 'xml') {
      return code
        // Comments (VS Code: #6A9955)
        .replace(/(<!--[\s\S]*?-->)/g, '<span style="color: #6A9955; font-style: italic;">$1</span>')
        // Tags (VS Code: #569CD6)
        .replace(/(<\/?[a-zA-Z][^>]*>)/g, '<span style="color: #569CD6;">$1</span>')
        // Attributes (VS Code: #9CDCFE)
        .replace(/(\w+)=/g, '<span style="color: #9CDCFE;">$1</span>=')
        // Attribute values (VS Code: #CE9178)
        .replace(/="([^"]*)"/g, '="<span style="color: #CE9178;">$1</span>"');
    }
    
    if (language === 'json') {
      return code
        // Strings (VS Code: #CE9178)
        .replace(/("(?:[^"\\]|\\.)*")\s*:/g, '<span style="color: #9CDCFE;">$1</span>:')
        .replace(/:\s*("(?:[^"\\]|\\.)*")/g, ': <span style="color: #CE9178;">$1</span>')
        // Numbers (VS Code: #B5CEA8)
        .replace(/:\s*(\d+\.?\d*)/g, ': <span style="color: #B5CEA8;">$1</span>')
        // Booleans (VS Code: #569CD6)
        .replace(/:\s*(true|false|null)/g, ': <span style="color: #569CD6;">$1</span>');
    }
    
    return code;
  };

  return (
    <div className="flex h-full bg-[#1e1e1e] text-[#d4d4d4] font-mono text-sm">
      <div className="line-numbers bg-[#1e1e1e] px-4 py-4 text-[#858585] text-right select-none overflow-hidden border-r border-[#3e3e42] min-w-[4rem]">
        {lineNumbers.map((num) => (
          <div key={num} className="leading-6 h-6">
            {num}
          </div>
        ))}
      </div>
      <div className="flex-1 relative">
        <LiveCursors cursors={collaborators} currentFile={fileName} />
        
        {/* Syntax highlighted background */}
        <div 
          ref={highlightRef}
          className="absolute inset-0 p-4 pointer-events-none whitespace-pre-wrap break-words font-mono text-sm leading-6 overflow-hidden"
          dangerouslySetInnerHTML={{ __html: syntaxHighlight(content) }}
          style={{ 
            color: 'transparent',
            background: 'transparent',
            zIndex: 1
          }}
        />
        
        {/* Transparent textarea for editing */}
        <textarea
          ref={textareaRef}
          className="absolute inset-0 w-full h-full p-4 bg-transparent resize-none outline-none font-mono text-sm leading-6 text-transparent caret-white selection:bg-[#264f78]"
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