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

  useEffect(() => {
    const lines = content.split('\n').length;
    setLineNumbers(Array.from({ length: Math.max(lines, 20) }, (_, i) => i + 1));
  }, [content]);

  const handleScroll = (e: React.UIEvent<HTMLTextAreaElement>) => {
    const lineNumbersEl = document.querySelector('.line-numbers');
    if (lineNumbersEl) {
      lineNumbersEl.scrollTop = e.currentTarget.scrollTop;
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
        .replace(/(fn|let|mut|pub|struct|impl|enum|match|if|else|while|for|loop|break|continue|return|use|mod|crate|super|self|const|static|unsafe|async|await|move|ref|dyn|where|type|trait|macro_rules!)/g, '<span class="text-purple-400">$1</span>')
        .replace(/(String|Vec|Option|Result|Box|Rc|Arc|HashMap|BTreeMap|i32|i64|u32|u64|f32|f64|bool|char|str|usize|isize)/g, '<span class="text-blue-400">$1</span>')
        .replace(/(println!|print!|dbg!|panic!|todo!|unimplemented!|unreachable!)/g, '<span class="text-green-400">$1</span>')
        .replace(/(".*?")/g, '<span class="text-yellow-400">$1</span>')
        .replace(/(\/\/.*$)/gm, '<span class="text-gray-500">$1</span>')
        .replace(/(\/\*[\s\S]*?\*\/)/g, '<span class="text-gray-500">$1</span>');
    }
    
    if (language === 'dart') {
      return code
        .replace(/(class|extends|implements|with|abstract|static|final|const|var|dynamic|void|int|double|String|bool|List|Map|Set|Future|Stream|async|await|yield|import|export|library|part|show|hide|as|if|else|for|while|do|switch|case|default|break|continue|return|try|catch|finally|throw|rethrow|assert)/g, '<span class="text-purple-400">$1</span>')
        .replace(/(Widget|StatelessWidget|StatefulWidget|State|BuildContext|MaterialApp|Scaffold|AppBar|Container|Column|Row|Text|Button|FloatingActionButton)/g, '<span class="text-blue-400">$1</span>')
        .replace(/(@override|@required|@deprecated)/g, '<span class="text-orange-400">$1</span>')
        .replace(/(".*?"|'.*?')/g, '<span class="text-yellow-400">$1</span>')
        .replace(/(\/\/.*$)/gm, '<span class="text-gray-500">$1</span>')
        .replace(/(\/\*[\s\S]*?\*\/)/g, '<span class="text-gray-500">$1</span>');
    }
    
    if (language === 'kotlin') {
      return code
        .replace(/(class|interface|object|enum|data|sealed|abstract|open|final|override|private|protected|public|internal|fun|val|var|const|lateinit|lazy|by|delegate|if|else|when|for|while|do|try|catch|finally|throw|return|break|continue|import|package)/g, '<span class="text-purple-400">$1</span>')
        .replace(/(String|Int|Long|Float|Double|Boolean|Char|Byte|Short|Any|Unit|Nothing|List|MutableList|Set|MutableSet|Map|MutableMap|Array|IntArray)/g, '<span class="text-blue-400">$1</span>')
        .replace(/(@\w+)/g, '<span class="text-orange-400">$1</span>')
        .replace(/(".*?")/g, '<span class="text-yellow-400">$1</span>')
        .replace(/(\/\/.*$)/gm, '<span class="text-gray-500">$1</span>')
        .replace(/(\/\*[\s\S]*?\*\/)/g, '<span class="text-gray-500">$1</span>');
    }
    
    if (language === 'typescript' || language === 'javascript') {
      return code
        .replace(/(import|export|from|default|const|let|var|function|class|interface|type|enum|namespace|async|await|return|if|else|for|while|do|switch|case|break|continue|try|catch|finally|throw|new|this|super|extends|implements|public|private|protected|static|readonly|abstract)/g, '<span class="text-purple-400">$1</span>')
        .replace(/(string|number|boolean|object|undefined|null|void|any|unknown|never|Array|Promise|React|Component|useState|useEffect|useCallback|useMemo|StyleSheet|View|Text|TouchableOpacity|SafeAreaView|ScrollView)/g, '<span class="text-blue-400">$1</span>')
        .replace(/(@\w+)/g, '<span class="text-orange-400">$1</span>')
        .replace(/(".*?"|'.*?'|`.*?`)/g, '<span class="text-yellow-400">$1</span>')
        .replace(/(\/\/.*$)/gm, '<span class="text-gray-500">$1</span>')
        .replace(/(\/\*[\s\S]*?\*\/)/g, '<span class="text-gray-500">$1</span>');
    }
    
    if (language === 'java') {
      return code
        .replace(/(public|private|protected|static|final|abstract|class|interface|extends|implements|import|package|new|this|super|return|if|else|for|while|do|switch|case|break|continue|try|catch|finally|throw|synchronized|volatile|transient|native|strictfp)/g, '<span class="text-purple-400">$1</span>')
        .replace(/(String|int|long|float|double|boolean|char|byte|short|void|Object|List|Map|Set|Array|ArrayList|HashMap|HashSet|Intent|Bundle|Activity|Application|Context|View|TextView|Button|LinearLayout|RelativeLayout)/g, '<span class="text-blue-400">$1</span>')
        .replace(/(@\w+)/g, '<span class="text-orange-400">$1</span>')
        .replace(/(".*?")/g, '<span class="text-yellow-400">$1</span>')
        .replace(/(\/\/.*$)/gm, '<span class="text-gray-500">$1</span>')
        .replace(/(\/\*[\s\S]*?\*\/)/g, '<span class="text-gray-500">$1</span>');
    }
    
    if (language === 'objective-c') {
      return code
        .replace(/(@interface|@implementation|@end|@property|@synthesize|@dynamic|@protocol|@optional|@required|@class|@import|#import|#include|if|else|for|while|do|switch|case|break|continue|return|static|const|extern|typedef|struct|enum|union)/g, '<span class="text-purple-400">$1</span>')
        .replace(/(NSString|NSArray|NSDictionary|NSNumber|NSObject|UIView|UIViewController|UIApplication|UIWindow|BOOL|NSInteger|CGFloat|CGRect|CGPoint|CGSize|id|void|int|float|double|char|long|short)/g, '<span class="text-blue-400">$1</span>')
        .replace(/(@".*?")/g, '<span class="text-yellow-400">$1</span>')
        .replace(/(\/\/.*$)/gm, '<span class="text-gray-500">$1</span>')
        .replace(/(\/\*[\s\S]*?\*\/)/g, '<span class="text-gray-500">$1</span>');
    }
    
    return code;
  };

  return (
    <div className="flex h-full bg-gray-900 text-gray-100 font-mono text-sm">
      <div className="line-numbers bg-gray-800 px-4 py-4 text-gray-500 text-right select-none overflow-hidden border-r border-gray-700 min-w-[4rem]">
        {lineNumbers.map((num) => (
          <div key={num} className="leading-6">
            {num}
          </div>
        ))}
      </div>
      <div className="flex-1 relative">
        <LiveCursors cursors={collaborators} currentFile={fileName} />
        <textarea
          ref={textareaRef}
          className="w-full h-full p-4 bg-transparent resize-none outline-none font-mono text-sm leading-6 text-white"
          value={content}
          onChange={handleContentChange}
          onSelect={handleCursorMove}
          onScroll={handleScroll}
          spellCheck={false}
          placeholder={`Start coding in ${fileName}...`}
          style={{
            tabSize: 4,
          }}
        />
        <div 
          className="absolute inset-0 p-4 pointer-events-none whitespace-pre-wrap break-words font-mono text-sm leading-6 overflow-hidden"
          dangerouslySetInnerHTML={{ __html: syntaxHighlight(content) }}
          style={{ color: 'transparent' }}
        />
      </div>
    </div>
  );
};

export default CodeEditor;