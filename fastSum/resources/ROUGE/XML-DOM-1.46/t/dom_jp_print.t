BEGIN {print "1..3\n";}
END {print "not ok 1\n" unless $loaded;}
use XML::DOM;
use utf8;
$loaded = 1;
print "ok 1\n";

#Test 2
sub charRef2U8{
my $charRef = shift;
my $u8;
$charRef = pack("H*",sprintf("%x",$charRef));
  for (my $iLen = 0;$charRef ne "";$charRef = substr($charRef,$iLen)){
    if($charRef =~ /^\x00([\x00-\x7F])/){
      $iLen = 2;
      $u8 .= $1;
    }elsif($charRef =~ /^\x00([\x80-\xFF])/){
      $iLen = 2;
      $u8 .= pack("v@",
        (ord("\xC0")|
         ((ord($1) & 192) >> 6)));
      $u8 .= pack("v@",(ord("\x80")| (ord($2) & 63)));
    }elsif($charRef =~ /^([\x01-\x07])([\x00-\xFF])/){
      $iLen = 2;
      $u8 .= pack("v@",
        (ord("\xC0")|
         ((ord($1) & 7) << 2) |
         ((ord($2) & 192) >> 6)));
      $u8 .= pack("v@",(ord("\x80")| (ord($2) & 63)));
    }elsif($charRef =~ /^([\x08-\xD7])([\x00-\xFF])/){
      $iLen = 2;
      $u8 .= pack("v@",(ord("\xE0") | ((ord($1) & 240) >> 4)));
      $u8 .= pack("v@",(ord("\x80") |
        ((ord($1) & 15) << 2) |
        ((ord($2) & 192) >> 6)));
      $u8 .= pack("v@",(ord("\x80")| (ord($2) & 63)));
    }elsif($charRef =~ /^([\xD8-\xDB])([\x00-\xFF])([\xDC-\xDF])([\x00-\xFF])/){
      $iLen = 4;
      $u8 .= pack("v@",(ord("\xF4") |ord($1) & 3));
      $u8 .= pack("v@",(ord("\x80") |((ord($2) & 252)>> 2)));
      $u8 .= pack("v@",(ord("\x80") |
        ((ord($2) & 3) << 4) |
        ((ord($3) & 3) << 2) |
        ((ord($4) & 192) >> 6)));
      $u8 .= pack("v@",(ord("\x80") | (ord($4) & 63)));
    }elsif($charRef =~ /^([\xE0-\xFF])([\x00-\xFF])/){
      $iLen = 2;
      $u8 .= pack("v@",(ord("\xE0") | ((ord($1) & 240) >> 4)));
      $u8 .= pack("v@",(ord("\x80") |
        ((ord($1) & 15) << 2) |
        ((ord($2) & 192) >> 6)));
      $u8 .= pack("v@",(ord("\x80")| (ord($2) & 63)));
    }else{
      die "can\'t convert!\n";
    }
  }
  return $u8;
}

my $str = <<END;
<?xml version="1.0" standalone="yes"?>
<!DOCTYPE 文書 [
 <!ENTITY はっ "あはははむは">
 <!ELEMENT 文書 (ビーバス|バットヘッド)*>
 <!ELEMENT ビーバス (#PCDATA)>
 <!ELEMENT バットヘッド (#PCDATA)>
]>
<文書>
 <ビーバス>
おい、バットヘッド！
 </ビーバス>
 <バットヘッド>
なんだい、ビーバス
 </バットヘッド>  
 <ビーバス>
おまえ屁こいただろ &はっ;
 </ビーバス>
 <バットヘッド>
&はっ; そのとぉり &はっ;
 </バットヘッド>  
</文書>
END

my $parser = new XML::DOM::Parser (NoExpand => 1);
my $doc = $parser->parse ($str);
my $out = $doc->toString;
$out =~ tr/\012/\n/;
$out =~ s/(\&\#(\d+);)/sprintf("%s",charRef2U8($2))/eg;

if ($out ne $str)
{
    print "not ";
}
print "ok 2\n";

$str = $doc->getElementsByTagName("バットヘッド")->item(0)->toString;
$str =~ tr/\012/\n/;
$str =~ s/(\&\#(\d+);)/sprintf("%s",charRef2U8($2))/eg;

if ($str ne "<バットヘッド>\nなんだい、ビーバス\n </バットヘッド>")
{
    print "not ";
}
print "ok 3\n";
